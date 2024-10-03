import pathlib

import numpy as np
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from pytorch_msssim import ms_ssim
from FTDDM_MultiRes.utils import logs
import logging
from FTDDM_MultiRes.cInvNet_MRI_LearnablePrior import cInvNet_MRI_LearnablePrior
from FTDDM_MultiRes.MRSI_dataset_flow import load_data, RandomDownscale_function, RandomFlip_function, RandomShift_function, Met_Sampler
import time


def train_epoch(args, epoch, model, data_loader, optimizer, writer, global_step, diffusion):
    model.train()
    running_loss_nll, running_loss_logpz, running_loss_logdet, running_loss_L1, running_loss_ssim = 0, 0, 0, 0, 0
    total_data = len(data_loader)
    for iter, data in enumerate(data_loader):
        T1, flair, met_HR, data_max, Patient, sli, metname = data
        T1 = T1.float().cuda()
        flair = flair.float().cuda()
        met_HR = met_HR.float().cuda()

        if epoch == 0 and iter == 0:
            logging.info('--+' * 10)
            logging.info(f'T1 = {T1.shape}')
            logging.info(f'flair = {flair.shape}')
            logging.info(f'met_HR = {met_HR.shape} ')
            logging.info('--+' * 10)

        T1, flair, met_HR = RandomFlip_function(T1, flair, met_HR)
        T1, flair, met_HR = RandomShift_function(T1, flair, met_HR)
        met_LR, lowRes = RandomDownscale_function(met_HR, args.low_resolution)

        nonzero_mask = (met_HR != 0).float()
        #print(T1.shape, flair.shape)
        T1 = F.interpolate(T1, size=(64, 64, 1), mode='trilinear', align_corners=True).squeeze(-1)
        flair = F.interpolate(flair, size=(64, 64, 1), mode='trilinear', align_corners=True).squeeze(-1)

        noise = torch.randn_like(met_HR)
        noise = torch.mul(noise, nonzero_mask)
        T_trunc = torch.tensor([args.T_trunc]).cuda()
        met_HR_noisy = diffusion.q_sample(met_HR, T_trunc, noise=noise).detach()
        #print(T_trunc, lowRes)
        if global_step == 0:
            with torch.no_grad():
                z, logdet = model(met_HR_noisy, nonzero_mask, met_LR, T1, flair, lowRes, metname, initialize=True, cal_jacobian=True)
            global_step += 1
            continue

        z, logdet = model(met_HR_noisy, nonzero_mask, met_LR, T1, flair, lowRes, metname, cal_jacobian=True)
        logdet += float(-np.log(256.) * met_HR.shape[1]*met_HR.shape[2]*met_HR.shape[3])
        logdet = logdet / float(np.log(2.)*met_HR.shape[1]*met_HR.shape[2]*met_HR.shape[3])
        logdet = -logdet.mean()

        z_ = [z_.view(z_.shape[0], -1) for z_ in z]
        z = torch.cat(z_, dim=1)
        prior_mean, prior_std = model.prior(met_LR)
        logpz = -0.5 * ((((z - prior_mean) / prior_std) ** 2) + torch.log(2 * np.pi * prior_std ** 2)).sum(-1)
        logpz = logpz / float(np.log(2.)*met_HR.shape[1]*met_HR.shape[2]*met_HR.shape[3])
        logpz = -logpz.mean()

        nll = logdet + logpz

        #guide loss
        loss_L1, loss_ssim = torch.tensor(0.0), torch.tensor(0.0)
        z_flat_guide = prior_mean
        if args.guide_weight > 0.0:
            z_sample_guide = reshape_noise(args, z_flat_guide, met_HR)
            outputs_back_guide = model(z_sample_guide, nonzero_mask, met_LR, T1, flair, lowRes, metname, rev=True)
            outputs_back_guide = torch.mul(outputs_back_guide, nonzero_mask)
            loss_L1 = F.l1_loss(outputs_back_guide, met_HR)
            loss_ssim = ssimloss(outputs_back_guide, met_HR)

        loss = nll + args.guide_weight * (0.16 * loss_L1 + 0.84 * loss_ssim)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss_nll += nll.item()
        running_loss_logpz += logpz.item()
        running_loss_logdet += logdet.item()
        running_loss_L1 += loss_L1.item()
        running_loss_ssim += loss_ssim.item()

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{total_data:4d}] '
                f'NLL Loss = {nll.item():.5g} '
                f'Logpz Loss = {logpz.item():.5g} '
                f'Logdet Loss = {logdet.item():.5g} '
                f'L1 Loss = {loss_L1.item():.5g} '
                f'SSIM Loss = {loss_ssim.item():.5g} '
            )
        global_step += 1

    nll = running_loss_nll / total_data
    logpz = running_loss_logpz / total_data
    logdet = running_loss_logdet / total_data
    loss_L1 = running_loss_L1 / total_data
    loss_ssim = running_loss_ssim / total_data
    if writer is not None:
        writer.add_scalar('Train/NLL', nll, epoch)
        writer.add_scalar('Train/Logpz', logpz, epoch)
        writer.add_scalar('Train/Logdet', logdet, epoch)
        writer.add_scalar('Train/L1', loss_L1, epoch)
        writer.add_scalar('Train/SSIM', loss_ssim, epoch)

    return loss, global_step


def valid(args, epoch, model, data_loader, writer, diffusion):
    model.eval()
    running_loss_nll, running_loss_logpz, running_loss_logdet, running_loss_L1, running_loss_ssim, running_loss = 0, 0, 0, 0, 0, 0
    total_data = len(data_loader)
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            T1, flair, met_HR, data_max, Patient, sli, metname = data
            T1 = T1.float().cuda()
            flair = flair.float().cuda()
            met_HR = met_HR.float().cuda()

            T1, flair, met_HR = RandomFlip_function(T1, flair, met_HR)
            T1, flair, met_HR = RandomShift_function(T1, flair, met_HR)
            met_LR, lowRes = RandomDownscale_function(met_HR, args.low_resolution)
            nonzero_mask = (met_HR != 0).float()

            T1 = F.interpolate(T1, size=(64, 64, 1), mode='trilinear', align_corners=True).squeeze(-1)
            flair = F.interpolate(flair, size=(64, 64, 1), mode='trilinear', align_corners=True).squeeze(-1)

            noise = torch.randn_like(met_HR)
            noise = torch.mul(noise, nonzero_mask)
            T_trunc = torch.tensor([args.T_trunc]).cuda()
            met_HR_noisy = diffusion.q_sample(met_HR, T_trunc, noise=noise)

            z, logdet = model(met_HR_noisy, nonzero_mask, met_LR, T1, flair, lowRes, metname, cal_jacobian=True)
            logdet += float(-np.log(256.) * met_HR.shape[1] * met_HR.shape[2] * met_HR.shape[3])
            logdet = logdet / float(np.log(2.) * met_HR.shape[1] * met_HR.shape[2] * met_HR.shape[3])
            logdet = -logdet.mean()

            z_ = [z_.view(z_.shape[0], -1) for z_ in z]
            z = torch.cat(z_, dim=1)
            prior_mean, prior_std = model.prior(met_LR)
            logpz = -0.5 * ((((z - prior_mean) / prior_std) ** 2) + torch.log(2 * np.pi * prior_std ** 2)).sum(-1)
            logpz = logpz / float(np.log(2.) * met_HR.shape[1] * met_HR.shape[2] * met_HR.shape[3])
            logpz = -logpz.mean()

            nll = logdet + logpz

            # guide loss
            loss_L1, loss_ssim = torch.tensor(0.0), torch.tensor(0.0)
            z_flat_guide = prior_mean
            if args.guide_weight > 0.0:
                z_sample_guide = reshape_noise(args, z_flat_guide, met_HR)
                outputs_back_guide = model(z_sample_guide, nonzero_mask, met_LR, T1, flair, lowRes, metname, rev=True)
                outputs_back_guide = torch.mul(outputs_back_guide, nonzero_mask)
                loss_L1 = F.l1_loss(outputs_back_guide, met_HR)
                loss_ssim = ssimloss(outputs_back_guide, met_HR)

            loss = nll + args.guide_weight * (0.16 * loss_L1 + 0.84 * loss_ssim)

            running_loss_nll += nll.item()
            running_loss_logpz += logpz.item()
            running_loss_logdet += logdet.item()
            running_loss_L1 += loss_L1.item()
            running_loss_ssim += loss_ssim.item()
            running_loss += loss.item()

        nll = running_loss_nll / total_data
        logpz = running_loss_logpz / total_data
        logdet = running_loss_logdet / total_data
        loss_L1 = running_loss_L1 / total_data
        loss_ssim = running_loss_ssim / total_data
        loss = running_loss / total_data

    if writer is not None:
        writer.add_scalar('Dev/NLL', nll, epoch)
        writer.add_scalar('Dev/Logpz', logpz, epoch)
        writer.add_scalar('Dev/Logdet', logdet, epoch)
        writer.add_scalar('Dev/L1', loss_L1, epoch)
        writer.add_scalar('Dev/SSIM', loss_ssim, epoch)
        writer.add_scalar('Dev/Loss_Total', loss, epoch)
    return loss


def reshape_noise(args, z_flat, met_HR):
    sizes = [(2 ** i, met_HR.shape[3] // 2 ** i, met_HR.shape[3] // 2 ** i) for i in range(1, args.down_num)]
    sizes.append((2 ** (args.down_num - 1) * 4, met_HR.shape[3] // 2 ** args.down_num, met_HR.shape[3] // 2 ** args.down_num))
    z_sample = []
    cur_dim = 0
    for z_shape in sizes:
        z_dim = np.prod(z_shape)
        this_z = z_flat[:, cur_dim: cur_dim + z_dim]
        this_z = this_z.view(z_flat.size(0), *z_shape)
        z_sample.append(this_z)
        cur_dim += z_dim
    return z_sample


def ssimloss(output, target):
    output_ = F.interpolate(output, mode='bicubic', size=(192, 192), align_corners=True)
    target_ = F.interpolate(target, mode='bicubic', size=(192, 192), align_corners=True)
    ssim_loss = 1 - ms_ssim(output_, target_, data_range=torch.max(target_) - torch.min(target_), size_average=True)
    return ssim_loss


def create_data_loaders(args):
    train_Dataset = load_data(patients=args.train_patients, transform=None, mode='train')
    met_sampler = Met_Sampler(batch_size=args.batch_size, length=train_Dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_Dataset, batch_sampler=met_sampler, num_workers=0)
    valid_Dataset = load_data(patients=args.valid_patients, transform=None, mode='valid')
    valid_loader = torch.utils.data.DataLoader(valid_Dataset, batch_size=1, shuffle=False, num_workers=0)
    return train_loader, valid_loader


def save_model(args, exp_dir, epoch, model, optimizer, valid_loss, best_valid_loss):
    logging.info('Saving trained model')

    ## create a models folder if not exists
    if not (args.exp_dir / 'flow_models').exists():
        (args.exp_dir / 'flow_models').mkdir(parents=True, exist_ok=True)

    if epoch % 100 == 99 or epoch == 0:
        torch.save({'epoch': epoch, 'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   f=str(exp_dir) + '/flow_models/epoch' + str(epoch) + '.pt')
        logging.info('Done saving model')

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save({'epoch': epoch, 'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   f=str(exp_dir) + '/flow_models/best_model.pt')
        logging.info('Done saving best model')
    return best_valid_loss


def build_model(args):
    model = cInvNet_MRI_LearnablePrior(channel_in=1, block_num=args.block_num, feature=args.feature, down_num=args.down_num).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9))
    return model, optimizer


def train_flow(args, diffusion):
    args.exp_dir = pathlib.Path(args.save_dir)

    logs.set_logger(str(args.exp_dir / 'train_flow.log'))
    logging.info('--' * 10)
    logging.info(
        '%s create log file %s' % (datetime.datetime.now().replace(microsecond=0), str(args.exp_dir / 'train_flow.log')))

    writer = SummaryWriter(log_dir=args.exp_dir / 'summary_flow')

    model, optimizer = build_model(args)

    logging.info('--' * 10)
    logging.info(args)
    logging.info('--' * 10)
    logging.info(model)
    logging.info('--' * 10)
    logging.info('Total parameters: %s' % sum(p.numel() for p in model.parameters()))
    logging.info('--' * 10)

    train_loader, valid_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    logging.info('--' * 10)
    start_training = datetime.datetime.now().replace(microsecond=0)
    logging.info('Start training at %s' % str(start_training))

    best_valid_loss = 1e9
    valid_loss = 1e10
    global_step = 0
    for epoch in range(0, args.num_epochs):
        logging.info('Current LR %s' % (scheduler.get_lr()[0]))
        torch.manual_seed(epoch)

        train_loss, global_step = train_epoch(args, epoch, model, train_loader, optimizer, writer, global_step, diffusion)

        if epoch > args.num_epochs * args.valid_epoch:
            valid_loss = valid(args, epoch, model, valid_loader, writer, diffusion)
        best_valid_loss = save_model(args, args.exp_dir, epoch, model, optimizer, valid_loss, best_valid_loss)

        scheduler.step(epoch)
        logging.info('Epoch: %s Reduce LR to: %s' % (epoch, scheduler.get_lr()[0]))

    writer.close()