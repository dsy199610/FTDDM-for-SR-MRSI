"""
Train a FTDDM model on images.
"""

import argparse

from FTDDM_MultiRes import dist_util, logger
from FTDDM_MultiRes.MRSI_dataset import load_data
from FTDDM_MultiRes.resample import create_named_schedule_sampler
from FTDDM_MultiRes.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from FTDDM_MultiRes.train_util import TrainLoop
import torch
import random
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import shutil
import sys
import pathlib
from FTDDM_MultiRes.train_flow import train_flow


def main():
    args = create_argparser().parse_args()
    args.save_dir = args.save_dir + '/testpatients_' + str(args.test_patients[0]) + '_' + str(args.test_patients[1]) + \
                    '_' + str(args.test_patients[2]) + '_' + str(args.test_patients[3]) + '_' + str(args.test_patients[4])
    dist_util.setup_dist()
    logger.configure(args.save_dir)

    save_python_script(args)
    writer = SummaryWriter(log_dir=pathlib.Path(args.save_dir + '/summary'))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, args.T_trunc)
    logger.log('Total parameters: %s' % sum(p.numel() for p in model.parameters()))

    logger.log("training flow")
    train_flow(args, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        patients=args.train_patients,
        batch_size=args.batch_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        T_trunc=args.T_trunc,
        diffusion=diffusion,
        data=data,
        low_resolution=args.low_resolution,
        class_cond=args.class_cond,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,  # exponential moving average
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        tb_writer=writer,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

    writer.close()


def save_python_script(args):
    ## copy training
    shutil.copyfile(sys.argv[0], str(args.save_dir) + '/' + sys.argv[0])
    ## copy models
    shutil.copytree('FTDDM_MultiRes', str(args.save_dir) + '/FTDDM_MultiRes')


def create_argparser():
    defaults = dict(
        train_patients=[17, 18, 19, 20, 22, 73, 78, 81, 84, 86, 91, 93, 96, 100, 104],
        valid_patients=[9, 12, 14, 15, 16],
        test_patients=[1, 2, 4, 6, 8],
        low_resolution=[4, 16],
        schedule_sampler="uniform",
        T_trunc=100,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=100000,
        batch_size=8, #8
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=100000,
        save_dir='checkpoints/FTDDM_MultiRes',

        # flow parameters
        num_epochs=500,
        valid_epoch=0.95,
        guide_weight=10.0,
        report_interval=10,
        down_num=3,
        feature=128,
        block_num=[12, 12, 12, 12],
        lr_step_size=100,
        lr_gamma=0.5,

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    main()