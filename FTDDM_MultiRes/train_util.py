import copy
import functools
import os
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.optim import Adam
import torch.nn.functional as F
from . import dist_util, logger
from .fp16_util import zero_grad
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from FTDDM_MultiRes.MRSI_dataset import RandomFlip_function, RandomShift_function, RandomDownscale_function


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        T_trunc,
        diffusion,
        data,
        low_resolution,
        class_cond,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        tb_writer,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.T_trunc = T_trunc
        self.diffusion = diffusion
        self.data = data
        self.low_resolution = low_resolution
        self.class_cond = class_cond
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.lr_current = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.tb_writter = tb_writer
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params

        self.opt = Adam(self.master_params, lr=self.lr, weight_decay=self.weight_decay) # originally AdamW, but version doesn't support
        self.ema_params = [copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))]

    def run_loop(self):
        while self.step < self.lr_anneal_steps:
            batch, prior, met_max, patient, sli, metname, cond = next(self.data)
            prior['T1'], prior['flair'] = prior['T1'].float(), prior['flair'].float()
            batch, prior['T1'], prior['flair'] = RandomFlip_function(batch, prior['T1'], prior['flair'])
            batch, prior['T1'], prior['flair'] = RandomShift_function(batch, prior['T1'], prior['flair'])
            batch_lr, lowRes = RandomDownscale_function(batch, self.low_resolution)
            if self.class_cond:
                cond['uf'] = 32.0 / th.from_numpy(np.array([lowRes])).expand(batch.shape[0]).float().cuda()
                #print(cond['uf'], cond['y'])
            self.run_step(batch, batch_lr, prior, cond, lowRes, metname)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, batch_lr, prior, cond, lowRes, metname):
        self.forward_backward(batch, batch_lr, prior, cond, lowRes, metname)
        self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, batch_lr, prior, cond, lowRes, metname):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_lr = batch_lr[i: i + self.microbatch].to(dist_util.dev())
            micro_prior = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in prior.items()
            }
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev()) # sample t

            nonzero_mask = (micro != 0).float()
            #print(micro_prior['T1'].shape, micro_prior['flair'].shape)
            micro_prior['T1'] = F.interpolate(micro_prior['T1'], size=(64, 64, 1), mode='trilinear', align_corners=True).squeeze(-1)
            micro_prior['flair'] = F.interpolate(micro_prior['flair'], size=(64, 64, 1), mode='trilinear', align_corners=True).squeeze(-1)
            
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                micro,
                micro_lr,
                micro_prior,
                t,
                self.T_trunc,
                self.step,
                lowRes,
                metname,
                nonzero_mask,
                model_kwargs=micro_cond,
            )

            losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})

            if self.step % self.log_interval == 0:
                plot_losses(self.tb_writter, self.step, {k: v * weights for k, v in losses.items()}, self.log_interval)

            loss.backward()

    def optimize_normal(self):
        #self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.step / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        self.lr_current = lr
        for param_group in self.opt.param_groups:
            param_group["lr"] = self.lr_current

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.global_batch)
        logger.logkv("lr", self.lr_current)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{self.step:06d}.pt"
                else:
                    filename = f"ema_{rate}_{self.step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        #save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def plot_losses(writer, step, losses, log_interval):
    if writer is not None:
        for key, values in losses.items():
            #print(key, values.mean().item(), step)
            writer.add_scalar('Train/'+key, values.mean().item(), step // log_interval)
