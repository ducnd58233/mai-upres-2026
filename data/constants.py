from __future__ import annotations

# Model / SR
SCALE_DEFAULT = 3
PIXEL_MAX = 255.0
PIXEL_MIN = 0.0

# Weight clipping (INT8-friendly training)
WEIGHT_CLIP_OTHER_DEFAULT = 2.0
WEIGHT_CLIP_REP_DEFAULT = 3.0

# DCT / PSNR
PSNR_EPS = 1e-12
PSNR_SATURATION = 99.0

# TFLite / PixelShuffle
DEPTH_TO_SPACE_BLOCK_DEFAULT = 3

# Export default input size (NCHW: 1, 3, H, W)
EXPORT_INPUT_HEIGHT_DEFAULT = 720
EXPORT_INPUT_WIDTH_DEFAULT = 1280
DEFAULT_EXPORT_INPUT_SHAPE = (
    1,
    3,
    EXPORT_INPUT_HEIGHT_DEFAULT,
    EXPORT_INPUT_WIDTH_DEFAULT,
)

# DIV2K path patterns (same dataset layout for all models)
DIV2K_TRAIN_HR_PATTERNS = [
    "DIV2K_train_HR/DIV2K_train_HR",
    "DIV2K_train_HR",
]
DIV2K_VALID_HR_PATTERNS = [
    "DIV2K_valid_HR/DIV2K_valid_HR",
    "DIV2K_valid_HR",
]
DIV2K_TRAIN_LR_PATTERNS_TEMPLATE = [
    "DIV2K_train_LR_bicubic/X{scale}",
    "DIV2K_train_LR_bicubic_X{scale}/DIV2K_train_LR_bicubic/X{scale}",
    "DIV2K_train_LR_bicubic_X{scale}/X{scale}",
]
DIV2K_VALID_LR_PATTERNS_TEMPLATE = [
    "DIV2K_valid_LR_bicubic/X{scale}",
    "DIV2K_valid_LR_bicubic_X{scale}/DIV2K_valid_LR_bicubic/X{scale}",
    "DIV2K_valid_LR_bicubic_X{scale}/X{scale}",
]
