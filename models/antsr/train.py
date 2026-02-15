from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.div2k import DIV2KPairX3, resolve_div2k_paths
from models.antsr.config import AntSRTrainConfig
from models.antsr.model import AntSR
from training.stage import (
    load_ckpt_weights,
    prepare_qat,
    strip_qat_to_clean_state,
    train_stage,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_antsr_training(
    config: AntSRTrainConfig,
    device: torch.device,
) -> str:
    logger.info("Device: %s", device)
    set_seed(config.seed)

    # Apply preset (use local vars; config is frozen)
    channels = config.channels
    n_rep = config.n_rep
    skip_mode = config.skip_mode
    concat_htr = config.concat_htr
    use_global_add = config.use_global_add
    if config.preset == "speed":
        channels = min(channels, 24)
        n_rep = max(n_rep, 4)
        use_global_add = False
        skip_mode = "concat_raw"
        concat_htr = "1x1_1x1"

    train_hr, train_lr, valid_hr, valid_lr = resolve_div2k_paths(
        config.data_root, scale=3
    )

    def make_train_loader(lr_patch: int) -> DataLoader:
        train_ds = DIV2KPairX3(
            train_hr, train_lr, train=True, lr_patch=lr_patch, augment=True, repeat=1
        )
        return DataLoader(
            train_ds,
            batch_size=config.batch,
            shuffle=True,
            num_workers=config.workers,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
        )

    val_ds = DIV2KPairX3(
        valid_hr, valid_lr, train=False, lr_patch=128, augment=False, repeat=1
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=(device.type == "cuda"),
    )

    model_cfg = {
        "scale": 3,
        "channels": channels,
        "n_rep": n_rep,
        "rep_use_bn": config.rep_use_bn,
        "rep_act_mode": config.rep_act_mode,
        "out_clamp_mode": config.out_clamp_mode,
        "skip_mode": skip_mode,
        "concat_htr": concat_htr,
        "use_global_add": use_global_add,
    }
    logger.info("MODEL CONFIG: %s", model_cfg)

    run_dir = os.path.join(config.out_dir, config.preset)
    os.makedirs(run_dir, exist_ok=True)

    model = AntSR(deploy=False, **model_cfg).to(device)

    if config.init_ckpt is not None:
        logger.info("Init from ckpt: %s", config.init_ckpt)
        load_ckpt_weights(model, config.init_ckpt, device)

    if config.epochs1 > 0:
        train_loader = make_train_loader(config.patch1)
        s1_best = train_stage(
            model,
            train_loader,
            val_loader,
            device,
            epochs=config.epochs1,
            base_lr=config.lr1,
            out_dir=run_dir,
            start_epoch=0,
            val_max=config.val_max,
            stage_tag="s1_fp32",
            grad_clip=config.grad_clip,
            scheduler_type=config.scheduler1,
            channel_shuffle=False,
            loss_mode=config.s1_loss,
            weight_clip=False,
            val_every=config.val_every,
        )
        logger.info("Load best Stage1 -> %s", s1_best)
        load_ckpt_weights(model, s1_best, device)

    if config.epochs2 > 0:
        train_loader = make_train_loader(config.patch2)
        start_ep = config.epochs1
        s2_best = train_stage(
            model,
            train_loader,
            val_loader,
            device,
            epochs=config.epochs2,
            base_lr=config.lr2,
            out_dir=run_dir,
            start_epoch=start_ep,
            val_max=config.val_max,
            stage_tag="s2_fp32",
            grad_clip=config.grad_clip,
            scheduler_type=config.scheduler2,
            channel_shuffle=config.channel_shuffle_s2s3,
            loss_mode=config.s2_loss,
            weight_clip=config.weight_clipping,
            wc_other=config.wc_other,
            wc_rep=config.wc_rep,
            val_every=config.val_every,
        )
        logger.info("Load best Stage2 -> %s", s2_best)
        load_ckpt_weights(model, s2_best, device)

        model.eval()
        model.switch_to_deploy()
        torch.save(
            {"model": model.state_dict(), "cfg": model_cfg},
            os.path.join(run_dir, "ckpt_best_s2_deploy.pt"),
        )
        logger.info("Saved: %s", os.path.join(run_dir, "ckpt_best_s2_deploy.pt"))

        model = AntSR(deploy=False, **model_cfg).to(device)
        load_ckpt_weights(model, s2_best, device)

    if config.qat and config.epochs3 > 0:
        train_loader = make_train_loader(config.patch3)
        start_ep = config.epochs1 + config.epochs2

        qat_on_deploy = False
        if config.deploy_before_qat:
            model.eval()
            model.switch_to_deploy()
            model.train()
            qat_on_deploy = True

        model = prepare_qat(model, backend=config.qat_backend).to(device)

        s3_best = train_stage(
            model,
            train_loader,
            val_loader,
            device,
            epochs=config.epochs3,
            base_lr=config.lr3,
            out_dir=run_dir,
            start_epoch=start_ep,
            val_max=config.val_max,
            stage_tag="s3_qat",
            grad_clip=config.grad_clip,
            scheduler_type=config.scheduler3,
            channel_shuffle=config.channel_shuffle_s2s3,
            loss_mode=config.s3_loss,
            weight_clip=config.weight_clipping,
            wc_other=config.wc_other,
            wc_rep=config.wc_rep,
            val_every=config.val_every,
        )
        logger.info("Best QAT ckpt: %s", s3_best)

        qat_ckpt = torch.load(s3_best, map_location="cpu", weights_only=False)
        qat_state = qat_ckpt["model"]

        clean = AntSR(deploy=qat_on_deploy, **model_cfg).eval()
        filtered = strip_qat_to_clean_state(clean, qat_state)
        clean.load_state_dict(filtered, strict=False)
        clean.switch_to_deploy()

        deploy_path = os.path.join(run_dir, "ckpt_best_s3_qat_deploy.pt")
        torch.save({"model": clean.state_dict(), "cfg": model_cfg}, deploy_path)
        logger.info("Saved: %s", deploy_path)

    logger.info("Done. run_dir: %s", run_dir)
    return run_dir
