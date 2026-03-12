#!/usr/bin/env bash
set -euo pipefail

# =========================
# EDIT THESE PATHS
# =========================
CKPT="${1:-ckpt_best_s3_qat_deploy.pt}"
OUT_DIR="${2:-artifacts_final}"
CALIB_LR_DIR="${3:-/path/to/DIV2K_train_LR_bicubic/X3}"
VALID_LR_DIR="${4:-/path/to/DIV2K_valid_LR_bicubic/X3}"
VALID_HR_DIR="${5:-/path/to/DIV2K_valid_HR}"
SANITY_IMG="${6:-/path/to/one_valid_lr_image.png}"
IDS_TXT="${7:-}"

mkdir -p "${OUT_DIR}"

FLOAT_TFLITE="${OUT_DIR}/model_float_nhwc.tflite"
PATCHED_FLOAT_TFLITE="${OUT_DIR}/model_float_nhwc_d2s.tflite"
INT8_TFLITE="${OUT_DIR}/model_int8.tflite"

python export_tflite_f32_nhwc.py \
  --ckpt "${CKPT}" \
  --out "${FLOAT_TFLITE}" \
  --h 720 \
  --w 1280

python patch_pixelshuffle_to_depth_to_space_permconv.py \
  --in_model "${FLOAT_TFLITE}" \
  --out_model "${PATCHED_FLOAT_TFLITE}" \
  --block 3

python check_tflite_io.py \
  --model "${PATCHED_FLOAT_TFLITE}"

python quantize_int8_static.py \
  --in_tflite "${PATCHED_FLOAT_TFLITE}" \
  --out_tflite "${INT8_TFLITE}" \
  --lr_dir "${CALIB_LR_DIR}" \
  --ids_txt "${IDS_TXT}" \
  --num_calib_images 800 \
  --crops_per_image 2 \
  --deterministic_calib \
  --metadata_json "${OUT_DIR}/quant_metadata.json"

python check_tflite_io.py \
  --model "${INT8_TFLITE}"

python list_float_tensors.py \
  --model "${INT8_TFLITE}"

python verify_submission_tflite.py \
  --model "${INT8_TFLITE}" \
  --input_h 720 \
  --input_w 1280 \
  --scale 3 \
  --strict_no_float_tensors

python sanity_compare_tflite_vs_pytorch.py \
  --tflite "${INT8_TFLITE}" \
  --pt_ckpt "${CKPT}" \
  --img "${SANITY_IMG}" \
  --h 720 \
  --w 1280

python eval_tflite_metrics.py \
  --model "${INT8_TFLITE}" \
  --lr_dir "${VALID_LR_DIR}" \
  --hr_dir "${VALID_HR_DIR}" \
  --num_images 100 \
  --crops_per_image 1 \
  --h 720 \
  --w 1280 \
  --deterministic \
  --compute_ssim \
  --save_json "${OUT_DIR}/eval_tflite_metrics.json"

echo "Done. Final INT8 model: ${INT8_TFLITE}"