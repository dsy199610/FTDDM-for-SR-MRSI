"""
Generate images from trained FTDDM model.
"""

import argparse
import pathlib
import numpy as np
import torch as th
import random
from FTDDM_MultiRes.utils import utils
from FTDDM_MultiRes import dist_util, logger
from FTDDM_MultiRes.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from FTDDM_MultiRes.cInvNet_MRI_LearnablePrior import cInvNet_MRI_LearnablePrior
from FTDDM_MultiRes.MRSI_dataset import load_testdata, RandomDownscale_function
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse
import matplotlib.pyplot as plt
import torch.nn.functional as F
import lpips

loss_fn_alex = lpips.LPIPS(net='alex').cuda()
loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()


def reshape_noise(args, z_flat, met_HR):
    sizes = [(2 ** i, met_HR.shape[3] // 2 ** i, met_HR.shape[3] // 2 ** i) for i in range(1, 3)]
    sizes.append((2 ** (3 - 1) * 4, met_HR.shape[3] // 2 ** 3, met_HR.shape[3] // 2 ** 3))
    z_sample = []
    cur_dim = 0
    for z_shape in sizes:
        z_dim = np.prod(z_shape)
        this_z = z_flat[:, cur_dim: cur_dim + z_dim]
        this_z = this_z.view(z_flat.size(0), *z_shape)
        z_sample.append(this_z)
        cur_dim += z_dim
    return z_sample


def compute_LPIPS(img, GT, loss):
    img = th.from_numpy(img).float().cuda()
    img = img.unsqueeze(0).unsqueeze(0)
    img = img.expand(1, 3, 64, 64)
    img = (img * 2 - img.max()) / img.max()

    GT = th.from_numpy(GT).float().cuda()
    GT = GT.unsqueeze(0).unsqueeze(0)
    GT = GT.expand(1, 3, 64, 64)
    GT = (GT * 2 - GT.max()) / GT.max()
    LPIPS = loss(img, GT).data
    LPIPS = LPIPS.cpu().numpy()[0][0][0][0]
    return LPIPS


def data_consistency(met_LR, output, lowRes):
    met_LR_cplx = th.cat((met_LR.unsqueeze(-1), th.zeros_like(met_LR.unsqueeze(-1))), -1)
    subkspace = utils.fft2(met_LR_cplx)
    output_cplx = th.cat((output.unsqueeze(-1), th.zeros_like(output.unsqueeze(-1))), -1)
    output_kspace = utils.fft2(output_cplx)
    d = output.shape[-1] // 2
    div = lowRes
    output_kspace[:, :, d - div:d + div, d - div:d + div, :] = subkspace[:, :, d - div:d + div, d - div:d + div, :]
    output_cplx = utils.ifft2(output_kspace)
    output = th.sqrt(output_cplx[:, :, :, :, 0] ** 2 + output_cplx[:, :, :, :, 1] ** 2)
    return output


def main():
    args = create_argparser().parse_args()

    log_dir = args.save_dir + '/testlog_respace' + str(args.timestep_respacing) + '_flowtemp' + str(args.gaussian_scale) + '_difftemp' + str(args.diff_temp) + '_lr' + str(args.low_resolution)
    logger.configure(dir=log_dir)

    img_dir = args.save_dir + '/images_respace' + str(args.timestep_respacing) + '_flowtemp' + str(args.gaussian_scale) + '_difftemp' + str(args.diff_temp) + '_lr' + str(args.low_resolution)
    pathlib.Path(img_dir).mkdir(parents=True, exist_ok=True)

    dist_util.setup_dist()

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.T_trunc != 0:
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
        model.to(dist_util.dev())
        model.eval()

    model_G = cInvNet_MRI_LearnablePrior(channel_in=1, block_num=args.block_num, feature=args.feature, down_num=args.down_num).cuda()
    model_G.load_state_dict(th.load(args.flow_dir)['model'])
    model_G.to(dist_util.dev())
    model_G.eval()

    dataloader = load_testdata(
        patients=args.test_patients,
        batch_size=args.batch_size,
        class_cond=args.class_cond
    )

    logger.log("sampling...")
    running_nrmse, running_psnr, running_ssim, running_lpips, running_lpips_vgg = [], [], [], [], []
    for iter, data in enumerate(dataloader):
        met_HR, prior, met_max, patient, sli, metname, cond = data
        met_HR = met_HR.cuda()
        if args.class_cond:
            cond['y'] = cond['y'].cuda()

        prior['T1'], prior['flair'] = prior['T1'].float().cuda(), prior['flair'].float().cuda()
        met_LR, lowRes = RandomDownscale_function(met_HR, args.low_resolution)

        if args.class_cond:
            cond['uf'] = 32.0 / th.from_numpy(np.array([lowRes])).expand(met_HR.shape[0]).float().cuda()

        nonzero_mask = (met_HR != 0).float()

        T1, flair = prior['T1'], prior['flair']
        prior['T1'] = F.interpolate(prior['T1'], size=(64, 64, 1), mode='trilinear', align_corners=True).squeeze(-1)
        prior['flair'] = F.interpolate(prior['flair'], size=(64, 64, 1), mode='trilinear', align_corners=True).squeeze(-1)
        with th.no_grad():
            T_trunc = th.tensor([args.T_trunc]).cuda()
            prior_mean, prior_std = model_G.prior(met_LR)
            z_flat = prior_mean + args.gaussian_scale * prior_std * th.randn_like(prior_mean)
            z_sample = reshape_noise(args, z_flat, met_HR)
            G_output = model_G(z_sample, nonzero_mask, met_LR, prior['T1'], prior['flair'], lowRes, metname, rev=True)
            G_output = th.mul(G_output, nonzero_mask)
            noise = th.randn_like(met_HR)
            noise = th.mul(noise, nonzero_mask)
            x_real = diffusion.q_sample(met_HR, T_trunc, noise=noise)

        if args.T_trunc == 0:
            sample = G_output
        else:
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                met_LR,
                prior,
                (met_LR.shape[0], 1, args.image_size, args.image_size),
                args.T_trunc,
                nonzero_mask,
                args.diff_temp,
                noise=G_output,
                clip_denoised=args.clip_denoised,
                model_kwargs=cond,  # model_kwargs,
            )

        if args.DataConsistency:
            sample = data_consistency(met_LR, sample, lowRes)


        nonzero_mask = (met_HR != 0).float()
        sample = th.mul(sample, nonzero_mask)
        sample = sample.clamp(0.0, 1.0)

        sli = sli.numpy()
        for i in range(sample.shape[0]):
            met_max_i = met_max[i].numpy()
            sample_i = sample[i].squeeze().cpu().numpy() * met_max_i
            met_LR_i = met_LR[i].squeeze().cpu().numpy() * met_max_i
            met_HR_i = met_HR[i].squeeze().cpu().numpy() * met_max_i
            G_output_i = G_output[i].squeeze().cpu().numpy() * met_max_i
            x_real_i = x_real[i].squeeze().cpu().numpy() * met_max_i
            T1_i = T1[i].squeeze().cpu().numpy()
            flair_i = flair[i].squeeze().cpu().numpy()

            NRMSE = normalized_root_mse(met_HR_i, sample_i, normalization='euclidean')
            PSNR = peak_signal_noise_ratio(met_HR_i, sample_i, data_range=met_HR_i.max() - met_HR_i.min())
            SSIM = structural_similarity(met_HR_i, sample_i, data_range=met_HR_i.max() - met_HR_i.min())
            LPIPS = compute_LPIPS(sample_i, met_HR_i, loss_fn_alex)
            LPIPS_vgg = compute_LPIPS(sample_i, met_HR_i, loss_fn_vgg)

            running_nrmse.append(NRMSE.item())
            running_psnr.append(PSNR.item())
            running_ssim.append(SSIM.item())
            running_lpips.append(LPIPS.item())
            running_lpips_vgg.append(LPIPS_vgg.item())

            logger.log(
                f'{patient[i]} slice={sli[i]} met={metname[i]} NRMSE={NRMSE} PSNR={PSNR} SSIM={SSIM} LPIPS={LPIPS} LPIPS VGG={LPIPS_vgg}')

            met_LR_plot = met_LR_i.repeat(3, axis=0).repeat(3, axis=1)
            met_HR_plot = met_HR_i.repeat(3, axis=0).repeat(3, axis=1)
            outputs_plot = sample_i.repeat(3, axis=0).repeat(3, axis=1)
            G_output_plot = G_output_i.repeat(3, axis=0).repeat(3, axis=1)
            x_real_plot = x_real_i.repeat(3, axis=0).repeat(3, axis=1)
            output_file = '%s/%s_slice%s_%s_NMSE%.3g_PSNR%.3g_SSIM%.3g_LPIPS%.3g_LPIPSVGG%.3g.png' % (
                img_dir, patient[i], sli[i], metname[i], NRMSE, PSNR, SSIM, LPIPS, LPIPS_vgg)
            diff = abs(outputs_plot - met_HR_plot) * 3.0
            plt.imsave(output_file, np.concatenate((met_LR_plot, met_HR_plot, outputs_plot, G_output_plot, x_real_plot, diff),
                                      axis=1), cmap='jet', vmin=met_HR_plot.min(), vmax=met_HR_plot.max())
            output_file = '%s/%s_slice%s_T1.png' % (img_dir, patient[i], sli[i])
            plt.imsave(output_file, np.concatenate((T1_i[:, :, 0], T1_i[:, :, 1], T1_i[:, :, 2]), axis=1), cmap='gray')
            output_file = '%s/%s_slice%s_flair.png' % (img_dir, patient[i], sli[i])
            plt.imsave(output_file, np.concatenate((flair_i[:, :, 0], flair_i[:, :, 1], flair_i[:, :, 2]), axis=1), cmap='gray')

            output_file = '%s/%s_slice%s_%s.npz' % (img_dir, patient[i], sli[i], metname[i])
            np.savez(output_file, met_LR=met_LR_i, output=sample_i, met_HR=met_HR_i)
            if metname[i] == 'Gln':
                output_file = '%s/%s_slice%s_MRI.npz' % (img_dir, patient[i], sli[i])
                np.savez(output_file, T1=T1_i, flair=flair_i)

    running_nrmse = np.asarray(running_nrmse)
    running_psnr = np.asarray(running_psnr)
    running_ssim = np.asarray(running_ssim)
    running_lpips = np.asarray(running_lpips)
    running_lpips_vgg = np.asarray(running_lpips_vgg)
    logger.info('NRMSE = %5g +- %5g' % (running_nrmse.mean(), running_nrmse.std()))
    logger.info('PSNR = %5g +- %5g' % (running_psnr.mean(), running_psnr.std()))
    logger.info('SSIM = %5g +- %5g' % (running_ssim.mean(), running_ssim.std()))
    logger.info('LPIPS = %5g +- %5g' % (running_lpips.mean(), running_lpips.std()))
    logger.info('LPIPS VGG = %5g +- %5g' % (running_lpips_vgg.mean(), running_lpips_vgg.std()))
    metric_dir = str(args.save_dir) + '/metrics_respace' + str(args.timestep_respacing) + '_flowtemp' + str(args.gaussian_scale) + '_difftemp' + str(args.diff_temp) + '_lr' + str(args.low_resolution)
    np.savez(metric_dir + '.npz', psnr=running_psnr, ssim=running_ssim, lpips=running_lpips, nrmse=running_nrmse, lpips_vgg=running_lpips_vgg)


def create_argparser():
    defaults = dict(
        test_patients=[91, 93, 96, 100, 104],
        low_resolution=4,
        clip_denoised=True,
        batch_size=32,
        use_ddim=False,
        DataConsistency=False,
        model_path="checkpoints/FTDDM_MultiRes/testpatients_91_93_96_100_104/ema_0.9999_100000.pt",
        save_dir='checkpoints/FTDDM_MultiRes/testpatients_91_93_96_100_104',
        flow_dir='checkpoints/FTDDM_MultiRes/testpatients_91_93_96_100_104/flow_models/best_model.pt',
        T_trunc=100,
        gaussian_scale=0.9,
        diff_temp=0.9,

        # flow parameters
        down_num=3,
        feature=128,
        block_num=[12, 12, 12, 12],
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    main()