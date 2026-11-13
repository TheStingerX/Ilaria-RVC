
import sys as _ilaria_sys
from pathlib import Path as _IlariaBootstrapPath
_ilaria_install_root = _IlariaBootstrapPath(__file__).resolve().parents[3]
_ilaria_sys.path.insert(0, str(_ilaria_install_root / "ilaria"))
_ilaria_sys.path.append(str(_ilaria_install_root))
import os
import sys
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

import datetime

from infer.lib.train import utils
from infer.lib.train.epoch_eta import EpochETA
from infer.lib.train.training_performance import configure_training_process
from ilaria_training_state import (
    training_signature, has_training_state, load_training_state,
    restore_runtime_state, epoch_stop_requested, gather_rng_states,
    save_training_state, share_boundary_error, distributed,
)
from ilaria_training_preview import TrainingPreview
from ilaria_training_targets import step_target, target_epoch_limit, StepETA

hps = utils.get_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))
from random import randint, shuffle

import torch

try:
    import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import

    if torch.xpu.is_available():
        from infer.modules.ipex import ipex_init
        from infer.modules.ipex.gradscaler import gradscaler_init
        from torch.xpu.amp import autocast

        GradScaler = gradscaler_init()
        ipex_init()
    else:
        from torch.cuda.amp import GradScaler, autocast
except Exception:
    from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from time import sleep
from time import time as ttime

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from infer.lib.infer_pack import commons
from infer.lib.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)

if hps.version == "v1":
    from infer.lib.infer_pack.models import MultiPeriodDiscriminator
    from infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid as RVC_Model_f0
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
    )
else:
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs768NSFsid as RVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
    )

from infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from ilaria_model_checkpoint import savee
from ilaria_vocoder import (apply_vocoder, build_refinegan_discriminator,
                            refinegan_mel_loss, validate_pretrain, weights_for_runtime)

global_step = 0


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        hours, remainder = divmod(max(0, int(elapsed_time)), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}h {minutes:02d}m  {seconds:02d}s"


@contextmanager
def frozen_discriminator(model):
    """Keep gradients through fake audio, without unused single-device D gradients."""
    # Keep the established reducer behavior for multi-GPU training.
    if isinstance(model, DDP):
        yield
        return
    parameters = tuple((parameter, parameter.requires_grad) for parameter in model.parameters())
    try:
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)


def main():
    n_gpus = torch.cuda.device_count()

    if torch.cuda.is_available() == False and torch.backends.mps.is_available() == True:
        n_gpus = 1
    if n_gpus < 1:
        # patch to unblock people without gpus. there is probably a better way.
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    children = []
    for i in range(n_gpus):
        subproc = mp.Process(
            target=run,
            args=(i, n_gpus, hps),
        )
        children.append(subproc)
        subproc.start()

    # A worker can fail before its peers reach a DDP collective. Poll all
    # children so the parent reports the real failure and releases those peers.
    while children:
        for child in list(children):
            child.join(timeout=0.2)
            if child.exitcode is None:
                continue
            children.remove(child)
            if child.exitcode != 0:
                for peer in children:
                    if peer.is_alive():
                        peer.terminate()
                for peer in children:
                    peer.join()
                raise SystemExit(child.exitcode if child.exitcode > 0 else 1)


def run(
    rank,
    n_gpus,
    hps,
):
    global global_step
    process_policy = configure_training_process()
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        logger.info("Training background policy: %s", process_policy)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer.add_scalar("training/run_started", 1, 0)
        writer.flush()
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    # A single device needs neither a process group nor DDP reducers.
    if n_gpus > 1:
        dist.init_process_group(
            backend="gloo", init_method="env://", world_size=n_gpus, rank=rank
        )
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        worker_init_fn=configure_training_process,
        prefetch_factor=8,
    )
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
        )
    selected_vocoder = getattr(hps.train, "vocoder", "HiFi-GAN")
    net_g = apply_vocoder(net_g, selected_vocoder, hps.version, hps.if_f0, hps.sample_rate)
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    net_d = (build_refinegan_discriminator(hps.model.use_spectral_norm)
             if selected_vocoder == "RefineGAN" else MultiPeriodDiscriminator(hps.model.use_spectral_norm))
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if n_gpus > 1:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            pass
        elif torch.cuda.is_available():
            net_g = DDP(net_g, device_ids=[rank])
            net_d = DDP(net_d, device_ids=[rank])
        else:
            net_g = DDP(net_g)
            net_d = DDP(net_d)

    signature = training_signature(hps, n_gpus)
    saved_state = None
    target_steps = step_target(hps)
    if has_training_state(hps.model_dir):
        # A corrupt managed checkpoint must never silently restart a model.
        saved_state = load_training_state(hps.model_dir, [net_g, net_d],
                                          [optim_g, optim_d], signature)
        epoch_str = saved_state["completed_epoch"] + 1
        global_step = saved_state["global_step"]
    else:
        # Older experiments used loose G/D files. Their iteration value means
        # a fully completed epoch, so continue at the following epoch.
        import glob
        legacy_g = glob.glob(os.path.join(hps.model_dir, "G_*.pth"))
        legacy_d = glob.glob(os.path.join(hps.model_dir, "D_*.pth"))
        if legacy_g or legacy_d:
            if not legacy_g or not legacy_d:
                raise ValueError("The old experiment is missing its G or D checkpoint. Restore the matching pair or use a new Model Name.")
            _, _, _, completed_d = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
            _, _, _, completed_g = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
            if completed_g != completed_d:
                raise ValueError("The saved generator and discriminator belong to different epochs.")
            epoch_str = int(completed_g) + 1
            global_step = int(completed_g) * len(train_loader)
            for optimizer in (optim_g, optim_d):
                for group in optimizer.param_groups:
                    group["initial_lr"] = hps.train.learning_rate
                    # ExponentialLR's initial step below restores the next
                    # epoch's learning rate, which old files did not record.
                    group["lr"] = hps.train.learning_rate * hps.train.lr_decay ** max(0, completed_g - 1)
        else:
            epoch_str, global_step = 1, 0
            if hps.pretrainG:
                if rank == 0:
                    logger.info("loaded pretrained %s", hps.pretrainG)
                target = net_g.module if hasattr(net_g, "module") else net_g
                pretrain = torch.load(hps.pretrainG, map_location="cpu", weights_only=True)
                validate_pretrain(pretrain, selected_vocoder)
                target.load_state_dict(weights_for_runtime(pretrain["model"]))
            if hps.pretrainD:
                if rank == 0:
                    logger.info("loaded pretrained %s", hps.pretrainD)
                target = net_d.module if hasattr(net_d, "module") else net_d
                pretrain = torch.load(hps.pretrainD, map_location="cpu", weights_only=True)
                target.load_state_dict(weights_for_runtime(pretrain["model"]))

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=-1 if saved_state is not None else epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=-1 if saved_state is not None else epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)
    if saved_state is not None:
        restore_runtime_state(saved_state, [scheduler_g, scheduler_d], scaler, rank)
    hps.resume_epoch = epoch_str
    hps.resume_batch_offset = saved_state.get("batch_offset", 0) if saved_state else 0
    if hps.resume_batch_offset and hps.if_cache_data_in_gpu:
        raise ValueError("Cache data to GPU must be No to resume a saved partial epoch.")
    epoch_limit = target_epoch_limit(hps, global_step, epoch_str, len(train_loader), hps.resume_batch_offset)
    hps.step_eta = StepETA(target_steps, global_step) if target_steps is not None and rank == 0 else None
    if rank == 0 and epoch_str > 1:
        logger.info("RESUMED! Epochs: %s | Total Steps: %s", epoch_str - 1, global_step)
    preview = TrainingPreview(hps, logger=logger) if rank == 0 else None

    cache = []
    epoch_eta = EpochETA(
        hps.total_epoch, completed_epochs=min(max(epoch_str - 1, 0), hps.total_epoch)
    ) if rank == 0 and target_steps is None else None
    for epoch in range(epoch_str, epoch_limit + 1):
        if rank == 0:
            boundary = train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
                cache,
                epoch_eta=epoch_eta,
                preview=preview,
                signature=signature,
            )
        else:
            boundary = train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                cache,
                signature=signature,
            )
        if boundary in ("stopped", "completed"):
            break
    if rank == 0:
        writer.flush()
        writer_eval.flush()
        writer.close()
        writer_eval.close()
    if distributed():
        dist.destroy_process_group()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, cache, epoch_eta=None, preview=None, signature=None
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    epoch_recorder = EpochRecorder()
    if epoch_eta is not None:
        epoch_eta.start_epoch()

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step
    target_steps = step_target(hps)
    resume_offset = getattr(hps, "resume_batch_offset", 0) if epoch == getattr(hps, "resume_epoch", 0) else 0
    completed_batches = resume_offset

    net_g.train()
    net_d.train()

    # Prepare data iterator
    if hps.if_cache_data_in_gpu == True:
        # Use Cache
        data_iterator = cache
        if cache == []:
            # Make new cache
            for batch_idx, info in enumerate(train_loader):
                # Unpack
                if hps.if_f0 == 1:
                    (
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                else:
                    (
                        phone,
                        phone_lengths,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                # Load on CUDA
                if torch.cuda.is_available():
                    phone = phone.cuda(rank, non_blocking=True)
                    phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                    if hps.if_f0 == 1:
                        pitch = pitch.cuda(rank, non_blocking=True)
                        pitchf = pitchf.cuda(rank, non_blocking=True)
                    sid = sid.cuda(rank, non_blocking=True)
                    spec = spec.cuda(rank, non_blocking=True)
                    spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                    wave = wave.cuda(rank, non_blocking=True)
                    wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
                # Cache on list
                if hps.if_f0 == 1:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                pitch,
                                pitchf,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
                else:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
        else:
            # Load shuffled cache
            shuffle(cache)
    else:
        # Loader
        data_iterator = enumerate(train_loader)

    # Run steps
    for batch_position, (batch_idx, info) in enumerate(data_iterator):
        if batch_position < resume_offset:
            continue
        log_step = rank == 0 and global_step % hps.train.log_interval == 0
        optim_g.zero_grad(set_to_none=True)
        optim_d.zero_grad(set_to_none=True)
        # Data
        ## Unpack
        if hps.if_f0 == 1:
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
        ## Load on CUDA
        if (hps.if_cache_data_in_gpu == False) and torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
            if hps.if_f0 == 1:
                pitch = pitch.cuda(rank, non_blocking=True)
                pitchf = pitchf.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)
            spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
            wave = wave.cuda(rank, non_blocking=True)
            # wave_lengths = wave_lengths.cuda(rank, non_blocking=True)

        # Calculate
        with autocast(enabled=hps.train.fp16_run):
            if hps.if_f0 == 1:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            else:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if hps.train.fp16_run == True:
                y_hat_mel = y_hat_mel.half()
            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat.detach())
            del fmap_r, fmap_g
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g, return_components=log_step
                )
        scaler.scale(loss_disc).backward()
        if log_step:
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)
        if not isinstance(net_d, DDP):
            optim_d.zero_grad(set_to_none=True)

        with frozen_discriminator(net_d):
            with autocast(enabled=hps.train.fp16_run):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
                with autocast(enabled=False):
                    loss_mel = (refinegan_mel_loss(wave, y_hat, hps.data.sampling_rate) * hps.train.c_mel / 3.0
                                if getattr(hps.train, "vocoder", "HiFi-GAN") == "RefineGAN"
                                else F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel)
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
            scaler.scale(loss_gen_all).backward()
            if log_step:
                scaler.unscale_(optim_g)
                grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if log_step:
                lr = optim_g.param_groups[0]["lr"]
                epoch_status = "Train Epoch: {} [{:.0f}%]".format(
                    epoch, 100.0 * batch_idx / len(train_loader)
                )
                logger.info(epoch_status)
                # Amor For Tensorboard display
                if loss_mel > 75:
                    loss_mel = 75
                if loss_kl > 9:
                    loss_kl = 9

                logger.info([global_step, lr])
                logger.info(
                    f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
                )
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl": loss_kl,
                    }
                )

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )
                writer.flush()
        global_step += 1
        completed_batches = batch_position + 1
        if target_steps is not None:
            if rank == 0 and getattr(hps, "step_eta", None) is not None:
                message = hps.step_eta.update(global_step, epoch,
                                             epoch if completed_batches == len(train_loader) else epoch - 1)
                if message:
                    logger.info(message)
            if global_step >= target_steps:
                break
    # /Run steps

    # Complete both learning-rate schedules before storing the boundary. The
    # resumed optimizer and scheduler therefore describe the NEXT epoch.
    partial_epoch = completed_batches < len(train_loader)
    completed_epoch = epoch - 1 if partial_epoch else epoch
    batch_offset = completed_batches if partial_epoch else 0
    if not partial_epoch:
        scheduler_g.step()
        scheduler_d.step()
    stopping = epoch_stop_requested(hps.model_dir, rank)
    final_epoch = global_step >= target_steps if target_steps is not None else epoch >= hps.total_epoch
    due = epoch % hps.save_every_epoch == 0 or stopping or final_epoch
    if due:
        rng_states = gather_rng_states()
        save_error = None
        if rank == 0:
            try:
                save_training_state(hps.model_dir, [net_g, net_d], [optim_g, optim_d],
                                    [scheduler_g, scheduler_d], scaler, completed_epoch,
                                    global_step, rng_states, signature,
                                    keep_latest=bool(hps.if_latest), batch_offset=batch_offset)
            except Exception as error:
                save_error = str(error)
        share_boundary_error(save_error, rank)
    if epoch % hps.save_every_epoch == 0 and rank == 0 and not partial_epoch:
        if rank == 0 and hps.save_every_weights == "1":
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    savee(
                        ckpt,
                        hps.sample_rate,
                        hps.if_f0,
                        hps.name + "_e%s_s%s" % (epoch, global_step),
                        epoch,
                        hps.version,
                        hps,
                    ),
                )
            )

    if rank == 0:
        if preview is not None and not stopping and not partial_epoch:
            preview.on_epoch(epoch, global_step, net_g)
    # A preview can take time. A Stop clicked while it is being rendered still
    # belongs to this boundary and must not trigger another complete epoch.
    after_preview_stop = epoch_stop_requested(hps.model_dir, rank)
    if after_preview_stop and not due:
        rng_states = gather_rng_states()
        save_error = None
        if rank == 0:
            try:
                save_training_state(hps.model_dir, [net_g, net_d], [optim_g, optim_d],
                                    [scheduler_g, scheduler_d], scaler, completed_epoch,
                                    global_step, rng_states, signature,
                                    keep_latest=bool(hps.if_latest), batch_offset=batch_offset)
            except Exception as error:
                save_error = str(error)
        share_boundary_error(save_error, rank)
    stopping = stopping or after_preview_stop
    if rank == 0:
        if epoch_eta is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            epoch_eta.finish_epoch(min(epoch, hps.total_epoch))
            logger.info("Training Model: {} ====> Epoch: {} || Total Steps: {} || Time taken: {} | {}".format(hps.name, epoch, global_step, epoch_recorder.record(), epoch_eta.summary(include_epoch_time=False)))
        else:
            logger.info("Training Model: {} ====> Epoch: {} || Total Steps: {} || Time taken: {}".format(hps.name, epoch, global_step, epoch_recorder.record()))
    # All ranks wait for the save/export result. No peer can start another
    # epoch after rank zero has accepted an orderly Stop and save request.
    if final_epoch:
        final_error = None
        if rank == 0:
            try:
                ckpt = net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()
                final_result = savee(ckpt, hps.sample_rate, hps.if_f0, hps.name,
                                     completed_epoch, hps.version, hps)
                if final_result != "Success.":
                    raise RuntimeError(str(final_result))
            except Exception as error:
                final_error = str(error)
        share_boundary_error(final_error, rank)
        if rank == 0:
            if target_steps is not None:
                logger.info("DONE! Step target reached | Total Steps: %s/%s", global_step, target_steps)
            logger.info("DONE! Training completed | Completed at: " + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        return "completed"
    if stopping:
        if distributed():
            dist.barrier()
        if rank == 0:
            logger.info("STOPPED! Training saved | Epoch: %s | Total Steps: %s | Saved at: %s",
                        epoch, global_step, datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        return "stopped"
    return None



if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
