# Mobile AI Workshop Super Resolution
## Prerequisites
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Docker](https://www.docker.com/) and Docker Compose
- Git

## Setup
### 1. Create Conda Environment

Create conda environment:

```bash
conda env create -f environment.yml
conda activate mai-upres
uv sync --all-groups
```

### 2. Install dependencies

```bash
uv add --group [name] <dep>
```

## Scripts

Run from project root with `python -m scripts.<name>` so the project is on `sys.path`. All model/export outputs go under `runs/models/` (train: `runs/models/<model-name>/`, export: `runs/models/export/`).

### 1. Download datasets (Kaggle)

```bash
uv run python -m scripts.download_kaggle_datasets <owner/slug>
# e.g. uv run python -m scripts.download_kaggle_datasets vivekanandabharupati/4k-images
```

### 2. Train

```bash
uv run python -m scripts.train --model <model-name> [--data-root <path>] [--out-dir <path>]
# e.g. uv run python -m scripts.train --model antsr
```

### 3. Convert PyTorch → TFLite

Output always in `runs/models/export/`; filename from checkpoint stem. Use `--model-type` for checkpoint format (e.g. `raw` needs `--input-shape`, `antsr` needs `--height`/`--width`).

```bash
uv run python -m scripts.torch_to_tflite --model <checkpoint.pt> --model-type <raw|antsr|...> [--input-shape 1 3 H W] [--height H --width W]
# raw:   --model-type raw --input-shape 1 3 224 224
# antsr: --model-type antsr --height 720 --width 1280
```

### 4. Eval

PyTorch: DIV2K valid. TFLite: any LR/HR dirs (needs `--lr-dir`, `--hr-dir`, `--scale`).

```bash
uv run python -m scripts.eval --model <model-name> --mode pytorch --checkpoint <path> --data-root <div2k-root>
uv run python -m scripts.eval --model <model-name> --mode tflite --tflite <path.tflite> --lr-dir <path> --hr-dir <path> --scale 3
```

### 5. Quantize to INT8 TFLite

Input: SavedModel dir or `.keras` file. Output: `runs/models/export/quantized_int8.tflite` by default. SavedModel from step 3: `runs/models/export/<stem>_tf/`.

```bash
uv run python -m scripts.quantize_tflite --saved-model <path> --calib-dir <image-dir> [--out-tflite <path>]
# or --keras-model <path> instead of --saved-model
```
