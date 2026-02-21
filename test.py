# export_png_multi.py
from pathlib import Path
import argparse
import pyarrow.parquet as pq

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

def iter_parquet_files(inp: str):
    p = Path(inp)
    # Nếu là thư mục -> lấy tất cả *.parquet (kể cả trong subfolder)
    if p.exists() and p.is_dir():
        return sorted(p.rglob("*.parquet"))
    # Nếu là file -> trả về file đó
    if p.exists() and p.is_file():
        return [p]
    # Nếu là glob pattern (vd: "data/*.parquet")
    return sorted(Path(".").glob(inp))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="dir / single parquet / glob, e.g. data_dir or 'data/*.parquet'")
    ap.add_argument("--out", default="all_png", help="output folder (flat)")
    ap.add_argument("--batch", type=int, default=2048, help="batch size")
    ap.add_argument("--skip-non-png", action="store_true", help="skip rows that are not PNG magic")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_parquet_files(args.inp)
    if not files:
        raise SystemExit(f"Không tìm thấy parquet nào từ --inp={args.inp}")

    total = 0
    skipped = 0

    for f in files:
        pf = pq.ParquetFile(str(f))
        prefix = f.stem  # tên file parquet (không đuôi)

        row_in_file = 0
        for batch in pf.iter_batches(batch_size=args.batch, columns=["image"]):
            img_struct = batch.column(0)           # struct: bytes, path
            bytes_arr = img_struct.field("bytes")
            path_arr  = img_struct.field("path")

            for i in range(batch.num_rows):
                b = bytes_arr[i].as_py()
                p = path_arr[i].as_py()

                if b is None:
                    row_in_file += 1
                    continue

                if args.skip_non_png and (not b.startswith(PNG_MAGIC)):
                    skipped += 1
                    row_in_file += 1
                    continue

                # Lấy tên ảnh gốc nếu có, nhưng luôn thêm row index để KHÔNG BAO GIỜ trùng tên
                stem = Path(p).stem if p else "img"
                out_name = f"{prefix}__{row_in_file:09d}__{stem}.png"
                (out_dir / out_name).write_bytes(b)

                total += 1
                row_in_file += 1

        print(f"[OK] {f} -> exported so far: {total}")

    print(f"\nDONE. Exported {total} PNGs to: {out_dir.resolve()}")
    if skipped:
        print(f"Skipped {skipped} non-PNG rows.")

if __name__ == "__main__":
    main()