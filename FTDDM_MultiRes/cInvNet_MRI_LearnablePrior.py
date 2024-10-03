import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride))

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, feature):
        super(DenseBlock, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(feature, channel_out, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return out


class ConditionalBlock(nn.Module):
    def __init__(self, scale_level, channel_in, channel_in_MRI, channel_out, feature):
        super(ConditionalBlock, self).__init__()

        channel_in = channel_in + channel_in_MRI + 1
        layers = []
        if scale_level == 0:
            layers.append(nn.Conv2d(channel_in, feature, kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        else:
            for i in range(scale_level):
                layers.append(nn.Conv2d(channel_in, feature, kernel_size=3, padding=1, stride=2))
                layers.append(nn.LeakyReLU(negative_slope=0.2))
                channel_in = feature

        layers.append(nn.Conv2d(feature, channel_out, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, c, c_LR, c_T1, c_flair):
        c = torch.cat((c, c_LR, c_T1, c_flair), dim=1)
        out = self.net(c)
        return out


class AffineCoupling(nn.Module):
    def __init__(self, channel_num, feature, scale_level, clamp=1.):
        super(AffineCoupling, self).__init__()

        self.channel_num = channel_num
        self.feature = feature
        self.clamp = clamp

        self.H = DenseBlock(self.channel_num, self.channel_num, self.feature)
        self.C = ConditionalBlock(scale_level, channel_in=1, channel_in_MRI=2, channel_out=self.channel_num // 2, feature=self.feature // 2)

    def forward(self, x, c, c_LR, c_T1, c_flair, rev=False):
        x1, x2 = x[:, :self.channel_num // 2, :, :], x[:, self.channel_num // 2:, :, :]

        if not rev:
            s_and_t = self.H(torch.cat((x1, self.C(c, c_LR, c_T1, c_flair)), dim=1))
            s, t = self.split(s_and_t)
            self.s = self.clamp * (torch.sigmoid(s) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + t
            y1 = x1
            y1, y2 = y2, y1
        else:
            x1, x2 = x2, x1
            s_and_t = self.H(torch.cat((x1, self.C(c, c_LR, c_T1, c_flair)), dim=1))
            s, t = self.split(s_and_t)
            self.s = self.clamp * (torch.sigmoid(s) * 2 - 1)
            y2 = (x2 - t).div(torch.exp(self.s))
            y1 = x1

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = self.s.view(self.s.shape[0], -1).sum(-1)
        else:
            jac = -self.s.view(self.s.shape[0], -1).sum(-1)

        return jac

    def split(self, x):
        xa = x[:, :self.channel_num // 2, :, :]
        xb = x[:, self.channel_num // 2:, :, :]
        return xa, xb


class AffineInjector(nn.Module):
    def __init__(self, channel_num, feature, scale_level, clamp=1.):
        super(AffineInjector, self).__init__()

        self.channel_num = channel_num
        self.feature = feature
        self.clamp = clamp

        self.H = DenseBlock(self.channel_num, self.channel_num * 2, self.feature)
        self.C = ConditionalBlock(scale_level, channel_in=1, channel_in_MRI=2, channel_out=self.channel_num, feature=self.feature // 2)

    def forward(self, x, c, c_LR, c_T1, c_flair, rev=False):
        if not rev:
            s_and_t = self.H(self.C(c, c_LR, c_T1, c_flair))
            s, t = self.split(s_and_t)
            self.s = self.clamp * (torch.sigmoid(s) * 2 - 1)
            y = x.mul(torch.exp(self.s)) + t
        else:
            s_and_t = self.H(self.C(c, c_LR, c_T1, c_flair))
            s, t = self.split(s_and_t)
            self.s = self.clamp * (torch.sigmoid(s) * 2 - 1)
            y = (x - t).div(torch.exp(self.s))
        return y

    def jacobian(self, x, rev=False):
        if not rev:
            jac = self.s.view(self.s.shape[0], -1).sum(-1)
        else:
            jac = -self.s.view(self.s.shape[0], -1).sum(-1)

        return jac

    def split(self, x):
        xa = x[:, :self.channel_num, :, :]
        xb = x[:, self.channel_num:, :, :]
        return xa, xb


class InvConv(nn.Module):

    def __init__(self, channel_num):
        super(InvConv, self).__init__()

        self.channel_num = channel_num
        self.W_size = [self.channel_num, self.channel_num, 1, 1]
        W_init = np.random.normal(0, 1, self.W_size[:-2])
        W_init = np.linalg.qr(W_init)[0].astype(np.float32)
        W_init = W_init.reshape(self.W_size)
        self.W = nn.Parameter(torch.tensor(W_init))

    def forward(self, x, c, c_LR, c_T1, c_flair, rev=False):
        if not rev:
            x = F.conv2d(x, self.W)
            return x
        else:
            inv_w = torch.inverse(self.W.squeeze().double()).float().view(self.W_size)
            x = F.conv2d(x, inv_w)
            return x

    def jacobian(self, x, c, rev=False):
        jac = x.shape[2] * x.shape[3] * torch.slogdet(self.W.squeeze())[1]
        jac = jac.view(1, 1).expand(x.shape[0], 1).squeeze()
        return jac


class Actnorm(nn.Module):

    def __init__(self, channel_num):
        super(Actnorm, self).__init__()

        self.channel_num = channel_num
        self.scale = torch.nn.Parameter(torch.zeros(1, self.channel_num, 1, 1), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(1, self.channel_num, 1, 1), requires_grad=True)

    def initialize(self, x):
        with torch.no_grad():
            bias = x.clone().mean(dim=(0, 2, 3), keepdim=True) * (-1.0)
            self.bias.data.copy_(bias.data)
            std = torch.sqrt(((x.clone() + bias) ** 2).mean(dim=(0, 2, 3), keepdim=True))
            logs = torch.log(1 / (std + 1e-6))
            self.scale.data.copy_(logs.data)
            print('initialized')

    def forward(self, x, c, c_LR,  c_T1, c_flair, rev=False):
        if not rev:
            self.s = self.scale.expand(x.shape[0], self.channel_num, 1, 1)
            x = x.mul(torch.exp(self.s))
            x = x + self.bias
            return x
        else:
            self.s = self.scale.expand(x.shape[0], self.channel_num, 1, 1)
            x = x - self.bias
            x = x.div(torch.exp(self.s))
            return x

    def jacobian(self, x, c, rev=False):
        jac = x.shape[2] * x.shape[3] * self.s.view(self.s.shape[0], -1).sum(-1)
        return jac


class Squeeze(nn.Module):

    def __init__(self):
        super(Squeeze, self).__init__()
        self.factor = 2

    def forward(self, x, c, c_LR, c_T1, c_flair, rev=False):
        n, c, h, w = x.size()

        if not rev:
            x = x.reshape(n, c, h // self.factor, self.factor, w // self.factor, self.factor)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()  # it seems this permutation works in nchw and nhwc
            x = x.reshape(n, c * self.factor * self.factor, h // self.factor, w // self.factor)
            return x
        else:
            x = x.reshape(n, c // (self.factor ** 2), self.factor, self.factor, h, w)
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
            x = x.reshape(n, c // (self.factor ** 2), h * self.factor, w * self.factor)
            return x

    def jacobian(self, x, rev=False):
        return 0.0


class Split(nn.Module):
    def __init__(self):
        super(Split, self).__init__()

    def forward(self, x, y=None, rev=False):
        n, c, h, w = x.size()
        if not rev:
            x1 = x[:, :c // 2, :, :]
            x2 = x[:, c // 2:, :, :]
            return x1, x2
        if rev:
            x = torch.cat([x, y], dim=1)
            return x


class cInvNet_MRI_LearnablePrior(nn.Module):
    def __init__(self, channel_in=1, block_num=[12, 12, 12, 12], feature=128, down_num=3):
        super(cInvNet_MRI_LearnablePrior, self).__init__()

        operations = []

        current_channel = channel_in
        for _ in range(block_num[0]):
            b = Actnorm(current_channel)
            operations.append(b)
            b = AffineInjector(current_channel, feature, 0)
            operations.append(b)
        for i in range(down_num):
            b = Squeeze()
            operations.append(b)
            current_channel *= 4
            b = Actnorm(current_channel)
            operations.append(b)
            b = InvConv(current_channel)
            operations.append(b)
            for j in range(block_num[i+1]):
                b = Actnorm(current_channel)
                operations.append(b)
                b = InvConv(current_channel)
                operations.append(b)
                b = AffineInjector(current_channel, feature, i+1)
                operations.append(b)
                b = AffineCoupling(current_channel, feature, i+1)
                operations.append(b)
            if i != down_num - 1:
                b = Split()
                operations.append(b)
                current_channel = current_channel // 2

        self.operations = nn.ModuleList(operations)

        dim = 4096
        self.prior_mu = nn.Parameter(torch.zeros((1, dim)).float(), requires_grad=True)
        self.prior_logstd = nn.Parameter(torch.zeros((1, dim)).float(), requires_grad=True)

    def forward(self, x, c, c_LR, c_T1, c_flair, lowRes, metname, initialize=False, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        if not rev:
            Z = []
            for op in self.operations:
                module_name = op.__class__.__name__
                if module_name == 'Split':
                    out, z = op.forward(out, rev)
                    Z.append(z)
                else:
                    if module_name == 'Actnorm' and initialize:
                        op.initialize(out)
                    out = op.forward(out, c, c_LR, c_T1, c_flair, rev)
                    if cal_jacobian:
                        jacobian += op.jacobian(out, rev)
            Z.append(out)
            out = Z
        else:
            out = [out[i] for i in range(len(out))]
            out_rev = out[-1]
            k = len(out) - 2
            for op in reversed(self.operations):
                module_name = op.__class__.__name__
                if module_name == 'Split':
                    out_rev = op.forward(out_rev, out[k], rev)
                    k = k - 1
                else:
                    out_rev = op.forward(out_rev, c, c_LR, c_T1, c_flair, rev)
            out = out_rev

        if cal_jacobian:
            return out, jacobian
        else:
            return out

    def prior(self, x):
        return self.prior_mu.expand(x.shape[0], 4096), torch.exp(self.prior_logstd).expand(x.shape[0], 4096) + 1e-6