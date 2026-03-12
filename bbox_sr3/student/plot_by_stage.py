import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def read_metrics(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing metrics.csv: {path}")

    df = pd.read_csv(path)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    for c in df.columns:
        if c in ("time", "stage", "best_key", "note"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    sort_cols = []
    if "stage" in df.columns:
        sort_cols.append("stage")
    if "time" in df.columns:
        sort_cols.append("time")
    elif "epoch" in df.columns:
        sort_cols.append("epoch")

    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort")

    return df.reset_index(drop=True)


def get_stage_order(df: pd.DataFrame):
    preferred = ["s1_fp32", "s2_fp32", "s3_qat"]
    existing = [s for s in preferred if s in df["stage"].dropna().unique().tolist()]
    others = sorted([s for s in df["stage"].dropna().unique().tolist() if s not in preferred])
    return existing + others


def choose_x(g: pd.DataFrame, prefer: str = "epoch"):
    prefer = prefer.lower()

    if prefer == "seq":
        return range(len(g)), "seq"

    if prefer == "time":
        if "time" in g.columns and g["time"].notna().any():
            return g["time"], "time"
        prefer = "epoch"

    if prefer == "epoch":
        if "epoch" in g.columns and g["epoch"].notna().any():
            e = g["epoch"].ffill()
            if (e.diff().fillna(0) >= 0).all():
                return g["epoch"], "epoch"
        return range(len(g)), "seq"

    return range(len(g)), "seq"


def finish_plot(out_path, title, xlabel, ylabel="value", has_line=True):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linewidth=0.3)
    if has_line:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_cols(out_path, x, xlabel, title, g: pd.DataFrame, cols, ylabel="value",
              deploy_value=None, deploy_label=None):
    plt.figure()
    has_line = False

    for c in cols:
        if c in g.columns and g[c].notna().any():
            plt.plot(x, g[c], marker="o", linewidth=1, markersize=3, label=c)
            has_line = True

    if deploy_value is not None and pd.notna(deploy_value):
        label = deploy_label if deploy_label else f"deploy={deploy_value:.4f}"
        plt.axhline(float(deploy_value), linestyle="--", linewidth=1.2, label=label)
        has_line = True

    finish_plot(out_path, title, xlabel, ylabel=ylabel, has_line=has_line)


def _parse_keyvals(text: str):
    """
    Parse key=value floats from a log line tail.
    Example: 'psnr_sr_rgb_sh0=30.1628 ssim=0.85'
    """
    out = {}
    for k, v in re.findall(r'([A-Za-z0-9_]+)=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', text):
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def read_deploy_info(run_dir: str):
    """
    Parse deploy-related info from train.log.

    Supported current patterns:
      [deploy-val/s2] psnr_sr_rgb_sh0=30.1234
      [deploy-val/s3] psnr_sr_rgb_sh0=30.1628

    Returns:
      deploy_info = {
        's2_fp32': {
            'deploy_last_time': ...,
            'deploy_best_psnr_sr_rgb_sh0': ...,
            'deploy_last_psnr_sr_rgb_sh0': ...,
            'deploy_entries': N,
            ... any best/last key found ...
        },
        's3_qat': {...}
      }
    """
    path = os.path.join(run_dir, "train.log")
    if not os.path.exists(path):
        return {}

    stage_alias = {
        "s1": "s1_fp32",
        "s2": "s2_fp32",
        "s3": "s3_qat",
    }

    deploy_rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r'^(.*?)\s+\|\s+INFO\s+\|\s+\[deploy-val/([^\]]+)\]\s+(.*)$', line.strip())
            if not m:
                continue

            t_str, short_stage, tail = m.groups()
            stage = stage_alias.get(short_stage.strip(), short_stage.strip())
            kv = _parse_keyvals(tail)

            row = {"stage": stage, "time": t_str}
            row.update(kv)
            deploy_rows.append(row)

    if not deploy_rows:
        return {}

    deploy_df = pd.DataFrame(deploy_rows)
    deploy_info = {}

    for stage, g in deploy_df.groupby("stage", sort=False):
        g = g.reset_index(drop=True)
        info = {
            "deploy_entries": len(g),
            "deploy_last_time": g.iloc[-1].get("time", None),
        }

        metric_cols = [c for c in g.columns if c not in ("stage", "time")]
        for c in metric_cols:
            s = pd.to_numeric(g[c], errors="coerce")
            if not s.notna().any():
                continue
            info[f"deploy_last_{c}"] = float(s.dropna().iloc[-1])
            info[f"deploy_best_{c}"] = float(s.max())

        deploy_info[stage] = info

    return deploy_info


def build_summary(df: pd.DataFrame, deploy_info: dict) -> pd.DataFrame:
    rows = []

    for st in get_stage_order(df):
        g = df[df["stage"] == st].copy()
        if g.empty or "psnr_sr_rgb_sh0" not in g.columns or not g["psnr_sr_rgb_sh0"].notna().any():
            continue

        idx = g["psnr_sr_rgb_sh0"].idxmax()
        best = g.loc[idx]

        row = {
            "stage": st,
            "best_epoch": best.get("epoch", None),
            "best_psnr_sr_rgb_sh0": best.get("psnr_sr_rgb_sh0", None),
            "best_psnr_sr_rgb_sh3": best.get("psnr_sr_rgb_sh3", None),
            "best_psnr_sr_y_sh0": best.get("psnr_sr_y_sh0", None),
            "best_psnr_sr_y_sh3": best.get("psnr_sr_y_sh3", None),
            "best_ssim_sr_rgb_sh0": best.get("ssim_sr_rgb_sh0", None),
            "best_ssim_sr_rgb_sh3": best.get("ssim_sr_rgb_sh3", None),
            "train_loss_at_best": best.get("train_loss", None),
            "lr_at_best": best.get("lr", None),
        }

        dep = deploy_info.get(st, {})
        for k, v in dep.items():
            row[k] = v

        rows.append(row)

    # also include stages that only have deploy info but no metrics rows
    existing_stages = {r["stage"] for r in rows}
    for st, dep in deploy_info.items():
        if st in existing_stages:
            continue
        row = {"stage": st}
        row.update(dep)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_stage_compare(df: pd.DataFrame, out_dir: str, deploy_info: dict):
    metrics = [
        ("psnr_sr_rgb_sh0", "compare_psnr_sr_rgb_sh0.png", "Compare stages | PSNR SR RGB sh0", "PSNR (dB)"),
        ("psnr_sr_rgb_sh3", "compare_psnr_sr_rgb_sh3.png", "Compare stages | PSNR SR RGB sh3", "PSNR (dB)"),
        ("ssim_sr_rgb_sh0", "compare_ssim_sr_rgb_sh0.png", "Compare stages | SSIM SR RGB sh0", "SSIM"),
        ("train_loss", "compare_train_loss.png", "Compare stages | Train loss", "loss"),
    ]

    stages = get_stage_order(df)

    for metric, filename, title, ylabel in metrics:
        plt.figure()
        has_line = False

        for st in stages:
            g = df[df["stage"] == st].copy()
            if metric in g.columns and g[metric].notna().any():
                x, _ = choose_x(g, prefer="epoch")
                plt.plot(x, g[metric], marker="o", linewidth=1, markersize=3, label=st)
                has_line = True

            # overlay deploy only for psnr_sr_rgb_sh0 if available
            if metric == "psnr_sr_rgb_sh0":
                dep = deploy_info.get(st, {})
                val = dep.get("deploy_best_psnr_sr_rgb_sh0", None)
                if val is not None:
                    plt.axhline(float(val), linestyle="--", linewidth=1.0, label=f"{st} deploy")
                    has_line = True

        finish_plot(os.path.join(out_dir, filename), title, "epoch/seq", ylabel=ylabel, has_line=has_line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--x", type=str, default="epoch", choices=["epoch", "time", "seq"])
    args = ap.parse_args()

    df = read_metrics(args.run_dir)

    if "stage" not in df.columns:
        raise RuntimeError("metrics.csv must contain column 'stage'")

    stages = get_stage_order(df)
    deploy_info = read_deploy_info(args.run_dir)

    base_out = os.path.join(args.run_dir, "plots_key")
    os.makedirs(base_out, exist_ok=True)

    summary = build_summary(df, deploy_info)
    summary_path = os.path.join(base_out, "summary_best.csv")
    summary.to_csv(summary_path, index=False)

    # save deploy-only summary too
    deploy_summary_path = os.path.join(base_out, "deploy_summary.csv")
    if deploy_info:
        pd.DataFrame(
            [{"stage": st, **info} for st, info in deploy_info.items()]
        ).to_csv(deploy_summary_path, index=False)
    else:
        pd.DataFrame(columns=["stage"]).to_csv(deploy_summary_path, index=False)

    for st in stages:
        g = df[df["stage"] == st].copy()
        if g.empty:
            continue

        out_dir = os.path.join(base_out, st)
        os.makedirs(out_dir, exist_ok=True)
        x, xlabel = choose_x(g, prefer=args.x)

        dep = deploy_info.get(st, {})
        deploy_psnr = dep.get("deploy_best_psnr_sr_rgb_sh0", None)

        # 1) PSNR chính + overlay deploy nếu có
        plot_cols(
            os.path.join(out_dir, "psnr_main.png"),
            x, xlabel, f"PSNR main | {st}", g,
            ["psnr_sr_rgb_sh0", "psnr_sr_rgb_sh3"],
            ylabel="PSNR (dB)",
            deploy_value=deploy_psnr,
            deploy_label=(f"deploy_psnr_sr_rgb_sh0={deploy_psnr:.4f}" if deploy_psnr is not None else None),
        )

        # 2) Delta so với bicubic
        plt.figure()
        has_line = False
        for sr_col, bi_col, label in [
            ("psnr_sr_rgb_sh0", "psnr_bi_rgb_sh0", "delta_rgb_sh0"),
            ("psnr_sr_rgb_sh3", "psnr_bi_rgb_sh3", "delta_rgb_sh3"),
        ]:
            if sr_col in g.columns and bi_col in g.columns:
                d = pd.to_numeric(g[sr_col], errors="coerce") - pd.to_numeric(g[bi_col], errors="coerce")
                if d.notna().any():
                    plt.plot(x, d, marker="o", linewidth=1, markersize=3, label=label)
                    has_line = True
        finish_plot(
            os.path.join(out_dir, "delta_vs_bicubic.png"),
            f"Delta PSNR vs Bicubic | {st}",
            xlabel,
            ylabel="delta (dB)",
            has_line=has_line
        )

        # 3) SSIM chính
        if "ssim_sr_rgb_sh0" in g.columns:
            plot_cols(
                os.path.join(out_dir, "ssim_main.png"),
                x, xlabel, f"SSIM main | {st}", g,
                ["ssim_sr_rgb_sh0", "ssim_sr_rgb_sh3"],
                ylabel="SSIM"
            )

        # 4) Loss
        plot_cols(
            os.path.join(out_dir, "loss.png"),
            x, xlabel, f"Train loss | {st}", g,
            ["train_loss"],
            ylabel="loss"
        )

        # 5) txt summary per-stage
        txt_path = os.path.join(out_dir, "stage_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            if "psnr_sr_rgb_sh0" in g.columns and g["psnr_sr_rgb_sh0"].notna().any():
                idx = g["psnr_sr_rgb_sh0"].idxmax()
                best = g.loc[idx]
                f.write(f"stage={st}\n")
                f.write(f"best_epoch={best.get('epoch', '')}\n")
                f.write(f"best_psnr_sr_rgb_sh0={best.get('psnr_sr_rgb_sh0', '')}\n")
                f.write(f"best_psnr_sr_rgb_sh3={best.get('psnr_sr_rgb_sh3', '')}\n")
                f.write(f"best_psnr_sr_y_sh0={best.get('psnr_sr_y_sh0', '')}\n")
                f.write(f"best_psnr_sr_y_sh3={best.get('psnr_sr_y_sh3', '')}\n")
                f.write(f"best_ssim_sr_rgb_sh0={best.get('ssim_sr_rgb_sh0', '')}\n")
                f.write(f"best_ssim_sr_rgb_sh3={best.get('ssim_sr_rgb_sh3', '')}\n")
                f.write(f"train_loss_at_best={best.get('train_loss', '')}\n")
                f.write(f"lr_at_best={best.get('lr', '')}\n")

            if dep:
                f.write("\n[deploy]\n")
                for k, v in dep.items():
                    f.write(f"{k}={v}\n")

        if dep:
            print(f"[OK] {st} -> {out_dir} | deploy: {dep}")
        else:
            print(f"[OK] {st} -> {out_dir}")

    compare_dir = os.path.join(base_out, "_compare")
    os.makedirs(compare_dir, exist_ok=True)
    plot_stage_compare(df, compare_dir, deploy_info)

    print(f"[OK] Summary -> {summary_path}")
    print(f"[OK] Deploy summary -> {deploy_summary_path}")
    print(f"Done -> {base_out}")


if __name__ == "__main__":
    main()