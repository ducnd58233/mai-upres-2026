from __future__ import annotations

import argparse

import torch

from configs import configure_logging
from models.antsr import AntSRTrainConfig, run_antsr_training
from training import get_default_out_dir

configure_logging()

DEFAULT_DATA_ROOT = "datasets/DIV2K"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train SR models. Use --model to select architecture.",
    )
    p.add_argument(
        "--model", type=str, required=True, choices=["antsr"], help="Model to train."
    )
    p.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help=f"Data root (default: {DEFAULT_DATA_ROOT}). For AntSR: DIV2K under this path.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: runs/models/<model-name>.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-max", type=int, default=0)
    p.add_argument("--val-every", type=int, default=1)

    # AntSR model
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--n-rep", type=int, default=4)
    p.add_argument(
        "--skip-mode",
        type=str,
        default="concat_raw",
        choices=["add", "add1x1", "concat_lr", "concat_raw"],
    )
    p.add_argument(
        "--concat-htr",
        type=str,
        default="3x3_3x3",
        choices=["3x3_3x3", "1x1_3x3", "1x1_1x1"],
    )
    p.add_argument(
        "--use-global-add", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--rep-use-bn", action="store_true")
    p.add_argument("--rep-act-mode", type=str, default="none", choices=["none", "relu"])
    p.add_argument(
        "--out-clamp-mode",
        type=str,
        default="minclip",
        choices=["min255", "minclip", "clamp_0_255", "none"],
    )

    # Stages
    p.add_argument("--epochs1", type=int, default=800)
    p.add_argument("--epochs2", type=int, default=200)
    p.add_argument("--epochs3", type=int, default=300)
    p.add_argument("--lr1", type=float, default=1e-3)
    p.add_argument("--lr2", type=float, default=2e-5)
    p.add_argument("--lr3", type=float, default=1e-5)
    p.add_argument("--patch1", type=int, default=128)
    p.add_argument("--patch2", type=int, default=128)
    p.add_argument("--patch3", type=int, default=128)
    p.add_argument("--s1-loss", type=str, default="l1", choices=["l1", "l2"])
    p.add_argument("--s2-loss", type=str, default="l2", choices=["l1", "l2"])
    p.add_argument("--s3-loss", type=str, default="dct", choices=["l1", "dct"])
    p.add_argument(
        "--scheduler1", type=str, default="cos_warmup", choices=["none", "cos_warmup"]
    )
    p.add_argument(
        "--scheduler2", type=str, default="step_halve", choices=["none", "step_halve"]
    )
    p.add_argument(
        "--scheduler3", type=str, default="step_halve", choices=["none", "step_halve"]
    )
    p.add_argument(
        "--channel-shuffle-s2s3", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--grad-clip", type=float, default=0.0)

    # QAT
    p.add_argument("--qat", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--qat-backend", type=str, default="qnnpack", choices=["qnnpack", "fbgemm"]
    )
    p.add_argument(
        "--deploy-before-qat", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--weight-clipping", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--wc-other", type=float, default=2.0)
    p.add_argument("--wc-rep", type=float, default=3.0)

    p.add_argument("--init-ckpt", type=str, default=None)
    p.add_argument(
        "--preset", type=str, default="balanced", choices=["balanced", "speed"]
    )

    return p.parse_args()


def _args_to_antsr_config(args: argparse.Namespace) -> AntSRTrainConfig:
    out_dir = (
        args.out_dir if args.out_dir is not None else get_default_out_dir(args.model)
    )
    return AntSRTrainConfig(
        data_root=args.data_root,
        out_dir=out_dir,
        seed=args.seed,
        batch=args.batch,
        workers=args.workers,
        val_max=args.val_max,
        val_every=args.val_every,
        channels=args.channels,
        n_rep=args.n_rep,
        skip_mode=args.skip_mode,
        concat_htr=args.concat_htr,
        use_global_add=args.use_global_add,
        rep_use_bn=args.rep_use_bn,
        rep_act_mode=args.rep_act_mode,
        out_clamp_mode=args.out_clamp_mode,
        epochs1=args.epochs1,
        epochs2=args.epochs2,
        epochs3=args.epochs3,
        lr1=args.lr1,
        lr2=args.lr2,
        lr3=args.lr3,
        patch1=args.patch1,
        patch2=args.patch2,
        patch3=args.patch3,
        s1_loss=args.s1_loss,
        s2_loss=args.s2_loss,
        s3_loss=args.s3_loss,
        scheduler1=args.scheduler1,
        scheduler2=args.scheduler2,
        scheduler3=args.scheduler3,
        channel_shuffle_s2s3=args.channel_shuffle_s2s3,
        grad_clip=args.grad_clip,
        qat=args.qat,
        qat_backend=args.qat_backend,
        deploy_before_qat=args.deploy_before_qat,
        weight_clipping=args.weight_clipping,
        wc_other=args.wc_other,
        wc_rep=args.wc_rep,
        init_ckpt=args.init_ckpt,
        preset=args.preset,
    )


def main() -> None:
    args = parse_args()

    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    if args.model == "antsr":
        config = _args_to_antsr_config(args)
        run_antsr_training(config, device)
    else:
        raise SystemExit(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
