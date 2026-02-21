# # #!/usr/bin/env python3
# # import argparse, glob, os, math, time
# # import numpy as np
# # from PIL import Image
# # import tflite_runtime.interpreter as tflite

# # def psnr(a, b, eps=1e-12):
# #     a = a.astype(np.float32)
# #     b = b.astype(np.float32)
# #     mse = np.mean((a - b) ** 2)
# #     if mse < eps:
# #         return 99.0
# #     return 10.0 * math.log10((255.0 ** 2) / mse)

# # def rgb2y_bt601(rgb):
# #     rgb = rgb.astype(np.float32)
# #     r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
# #     return 0.299 * r + 0.587 * g + 0.114 * b

# # def load_img(path):
# #     return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

# # def crop_border(x, crop):
# #     if crop <= 0:
# #         return x
# #     return x[crop:-crop, crop:-crop, ...]

# # def quantize_uint8_to_int8(x_u8, scale, zero_point):
# #     q = np.round(x_u8.astype(np.float32) / scale + zero_point).astype(np.int32)
# #     q = np.clip(q, -128, 127).astype(np.int8)
# #     return q

# # def dequant_int8_to_uint8(q_i8, scale, zero_point):
# #     x = (q_i8.astype(np.int32) - int(zero_point)) * float(scale)
# #     x = np.clip(np.round(x), 0, 255).astype(np.uint8)
# #     return x

# # def pad_reflect_hwc(img_hwc_u8, target_h, target_w):
# #     h, w = img_hwc_u8.shape[:2]
# #     if h > target_h or w > target_w:
# #         # center-crop if larger than model input
# #         y0 = max(0, (h - target_h) // 2)
# #         x0 = max(0, (w - target_w) // 2)
# #         img_hwc_u8 = img_hwc_u8[y0:y0+target_h, x0:x0+target_w, :]
# #         h, w = img_hwc_u8.shape[:2]

# #     pad_h = target_h - h
# #     pad_w = target_w - w
# #     if pad_h == 0 and pad_w == 0:
# #         return img_hwc_u8, 0, 0

# #     out = np.pad(img_hwc_u8, ((0,pad_h),(0,pad_w),(0,0)), mode="reflect")
# #     return out, pad_h, pad_w

# # def infer_one(itp, inp_u8_hwc, input_detail, output_detail, expect="auto"):
# #     in_shape = input_detail["shape"]
# #     in_dtype = input_detail["dtype"]
# #     in_scale, in_zp = input_detail.get("quantization", (1.0, 0))
# #     if len(in_shape) != 4:
# #         raise RuntimeError(f"Unexpected input rank: {in_shape}")

# #     # Determine model input layout + expected H,W
# #     if in_shape[1] == 3:
# #         layout_in = "NCHW"
# #         Hm, Wm = int(in_shape[2]), int(in_shape[3])
# #     else:
# #         layout_in = "NHWC"
# #         Hm, Wm = int(in_shape[1]), int(in_shape[2])

# #     # Pad/crop LR to model input size
# #     lr_pad, pad_h, pad_w = pad_reflect_hwc(inp_u8_hwc, Hm, Wm)

# #     # Build input tensor in the model's layout
# #     if layout_in == "NCHW":
# #         x = np.transpose(lr_pad, (2,0,1))[None, ...]  # 1,3,H,W
# #     else:
# #         x = lr_pad[None, ...]  # 1,H,W,3

# #     # Quantize / set tensor
# #     if in_dtype == np.int8:
# #         xq = quantize_uint8_to_int8(x, in_scale, in_zp)
# #         itp.set_tensor(input_detail["index"], xq)
# #     elif in_dtype == np.uint8:
# #         itp.set_tensor(input_detail["index"], x.astype(np.uint8))
# #     elif in_dtype == np.float32:
# #         itp.set_tensor(input_detail["index"], x.astype(np.float32))
# #     else:
# #         raise RuntimeError(f"Unsupported input dtype: {in_dtype}")

# #     itp.invoke()

# #     out = itp.get_tensor(output_detail["index"])
# #     out_dtype = output_detail["dtype"]
# #     out_scale, out_zp = output_detail.get("quantization", (1.0, 0))

# #     # Dequant to uint8 tensor (still 4D, model's output layout)
# #     if out_dtype == np.int8:
# #         out_u8 = dequant_int8_to_uint8(out, out_scale, out_zp)
# #     elif out_dtype == np.uint8:
# #         out_u8 = out.astype(np.uint8)
# #     elif out_dtype == np.float32:
# #         out_u8 = np.clip(np.round(out), 0, 255).astype(np.uint8)
# #     else:
# #         raise RuntimeError(f"Unsupported output dtype: {out_dtype}")

# #     if len(out_u8.shape) != 4:
# #         raise RuntimeError(f"Unexpected output rank: {out_u8.shape}")

# #     # Convert output to HWC uint8
# #     def out_to_hwc(out4):
# #         exp = expect.lower()
# #         if exp == "nhwc":
# #             if out4.shape[-1] == 3:
# #                 return out4[0]
# #             if out4.shape[1] == 3:
# #                 return np.transpose(out4[0], (1,2,0))
# #         elif exp == "nchw":
# #             if out4.shape[1] == 3:
# #                 return np.transpose(out4[0], (1,2,0))
# #             if out4.shape[-1] == 3:
# #                 return out4[0]

# #         # auto
# #         if out4.shape[-1] == 3:
# #             return out4[0]
# #         if out4.shape[1] == 3:
# #             return np.transpose(out4[0], (1,2,0))
# #         raise RuntimeError(f"Cannot interpret output layout: {out4.shape}")

# #     pred_hwc = out_to_hwc(out_u8)
# #     return pred_hwc, pad_h, pad_w, layout_in, (Hm, Wm)

# # def map_lr_to_hr_name(lr_name: str, scale: int) -> str:
# #     # DIV2K LR X3 often: 0801x3.png  -> HR: 0801.png
# #     stem, ext = os.path.splitext(lr_name)
# #     suf = f"x{scale}"
# #     if stem.endswith(suf):
# #         stem = stem[:-len(suf)]
# #     return stem + ext

# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--tflite", required=True, help="Path to .tflite model")
# #     ap.add_argument("--lr_dir", required=True, help="Folder of LR images (png/jpg)")
# #     ap.add_argument("--hr_dir", required=True, help="Folder of HR GT images")
# #     ap.add_argument("--glob", default="*.png", help="Glob for LR files")
# #     ap.add_argument("--num", type=int, default=0, help="0=all")
# #     ap.add_argument("--threads", type=int, default=4)
# #     ap.add_argument("--warmup", type=int, default=1)
# #     ap.add_argument("--crop", type=int, default=0, help="Crop border pixels before PSNR")
# #     ap.add_argument("--mode", choices=["rgb", "y"], default="rgb", help="PSNR on RGB or luma(Y)")
# #     ap.add_argument("--expect_out", choices=["auto","nhwc","nchw"], default="auto")
# #     ap.add_argument("--scale", type=int, default=3, help="Upscale factor (for cropping padded region + name mapping)")
# #     args = ap.parse_args()

# #     lr_files = sorted(glob.glob(os.path.join(args.lr_dir, args.glob)))
# #     if not lr_files:
# #         raise SystemExit(f"No files found: {os.path.join(args.lr_dir, args.glob)}")
# #     if args.num and args.num > 0:
# #         lr_files = lr_files[:args.num]

# #     itp = tflite.Interpreter(model_path=args.tflite, num_threads=args.threads)
# #     itp.allocate_tensors()
# #     in_det = itp.get_input_details()[0]
# #     out_det = itp.get_output_details()[0]

# #     # Warmup
# #     lr0 = load_img(lr_files[0])
# #     for _ in range(max(args.warmup, 0)):
# #         _pred0, *_ = infer_one(itp, lr0, in_det, out_det, expect=args.expect_out)

# #     scores = []
# #     t0 = time.time()
# #     for p in lr_files:
# #         lr_name = os.path.basename(p)
# #         hr_name = map_lr_to_hr_name(lr_name, args.scale)
# #         gt_path = os.path.join(args.hr_dir, hr_name)
# #         if not os.path.exists(gt_path):
# #             raise SystemExit(f"Missing GT: {gt_path}")

# #         lr = load_img(p)
# #         gt = load_img(gt_path)

# #         pred, pad_h, pad_w, layout_in, (Hm, Wm) = infer_one(itp, lr, in_det, out_det, expect=args.expect_out)

# #         # Remove SR area corresponding to padded LR bottom/right
# #         if pad_h > 0 or pad_w > 0:
# #             H_keep = pred.shape[0] - pad_h * args.scale
# #             W_keep = pred.shape[1] - pad_w * args.scale
# #             pred = pred[:H_keep, :W_keep, :]

# #         # Crop pred to GT size (GT is authoritative)
# #         pred = pred[:gt.shape[0], :gt.shape[1], :]

# #         # Optional border crop before PSNR
# #         pred_c = crop_border(pred, args.crop)
# #         gt_c   = crop_border(gt, args.crop)

# #         if pred_c.shape != gt_c.shape:
# #             raise SystemExit(
# #                 f"Shape mismatch for {lr_name}: pred {pred_c.shape} vs gt {gt_c.shape} "
# #                 f"(model_in={Hm}x{Wm}, LR={lr.shape[:2]}, pad={pad_h},{pad_w}, scale={args.scale}, hr_name={hr_name})"
# #             )

# #         if args.mode == "rgb":
# #             s = psnr(pred_c, gt_c)
# #         else:
# #             s = psnr(rgb2y_bt601(pred_c), rgb2y_bt601(gt_c))
# #         scores.append(s)

# #     dt = time.time() - t0
# #     print(f"Images: {len(scores)} | PSNR({args.mode}, crop={args.crop}): {sum(scores)/len(scores):.4f} dB | time={dt:.2f}s")

# # if __name__ == "__main__":
# #     main()




# #!/usr/bin/env python3
# import argparse, glob, os, math, time
# import numpy as np
# from PIL import Image
# import tflite_runtime.interpreter as tflite

# def psnr(a, b, eps=1e-12):
#     a = a.astype(np.float32)
#     b = b.astype(np.float32)
#     mse = np.mean((a - b) ** 2)
#     if mse < eps:
#         return 99.0
#     return 10.0 * math.log10((255.0 ** 2) / mse)

# def rgb2y_bt601(rgb):
#     rgb = rgb.astype(np.float32)
#     r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
#     return 0.299 * r + 0.587 * g + 0.114 * b

# def load_img(path):
#     return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

# def crop_border(x, crop):
#     if crop <= 0:
#         return x
#     return x[crop:-crop, crop:-crop, ...]

# def quantize_uint8_to_int8(x_u8, scale, zero_point):
#     q = np.round(x_u8.astype(np.float32) / scale + zero_point).astype(np.int32)
#     q = np.clip(q, -128, 127).astype(np.int8)
#     return q

# def dequant_int8_to_uint8(q_i8, scale, zero_point):
#     x = (q_i8.astype(np.int32) - int(zero_point)) * float(scale)
#     x = np.clip(np.round(x), 0, 255).astype(np.uint8)
#     return x

# def fit_lr_to_model(lr_hwc_u8, target_h, target_w):
#     """
#     If LR larger: center-crop (return offsets).
#     If LR smaller: reflect-pad bottom/right (return pad sizes).
#     """
#     h, w = lr_hwc_u8.shape[:2]
#     y0 = x0 = 0
#     pad_h = pad_w = 0

#     if h > target_h or w > target_w:
#         y0 = max(0, (h - target_h) // 2)
#         x0 = max(0, (w - target_w) // 2)
#         lr_hwc_u8 = lr_hwc_u8[y0:y0+target_h, x0:x0+target_w, :]
#         h, w = lr_hwc_u8.shape[:2]

#     if h < target_h or w < target_w:
#         pad_h = target_h - h
#         pad_w = target_w - w
#         lr_hwc_u8 = np.pad(lr_hwc_u8, ((0,pad_h),(0,pad_w),(0,0)), mode="reflect")

#     return lr_hwc_u8, y0, x0, pad_h, pad_w

# def infer_one(itp, inp_u8_hwc, input_detail, output_detail, expect="auto"):
#     in_shape = input_detail["shape"]
#     in_dtype = input_detail["dtype"]
#     in_scale, in_zp = input_detail.get("quantization", (1.0, 0))
#     if len(in_shape) != 4:
#         raise RuntimeError(f"Unexpected input rank: {in_shape}")

#     if in_shape[1] == 3:
#         layout_in = "NCHW"
#         Hm, Wm = int(in_shape[2]), int(in_shape[3])
#     else:
#         layout_in = "NHWC"
#         Hm, Wm = int(in_shape[1]), int(in_shape[2])

#     lr_fit, y0, x0, pad_h, pad_w = fit_lr_to_model(inp_u8_hwc, Hm, Wm)

#     if layout_in == "NCHW":
#         x = np.transpose(lr_fit, (2,0,1))[None, ...]
#     else:
#         x = lr_fit[None, ...]

#     if in_dtype == np.int8:
#         xq = quantize_uint8_to_int8(x, in_scale, in_zp)
#         itp.set_tensor(input_detail["index"], xq)
#     elif in_dtype == np.uint8:
#         itp.set_tensor(input_detail["index"], x.astype(np.uint8))
#     elif in_dtype == np.float32:
#         itp.set_tensor(input_detail["index"], x.astype(np.float32))
#     else:
#         raise RuntimeError(f"Unsupported input dtype: {in_dtype}")

#     itp.invoke()

#     out = itp.get_tensor(output_detail["index"])
#     out_dtype = output_detail["dtype"]
#     out_scale, out_zp = output_detail.get("quantization", (1.0, 0))

#     if out_dtype == np.int8:
#         out_u8 = dequant_int8_to_uint8(out, out_scale, out_zp)
#     elif out_dtype == np.uint8:
#         out_u8 = out.astype(np.uint8)
#     elif out_dtype == np.float32:
#         out_u8 = np.clip(np.round(out), 0, 255).astype(np.uint8)
#     else:
#         raise RuntimeError(f"Unsupported output dtype: {out_dtype}")

#     def out_to_hwc(out4):
#         exp = expect.lower()
#         if exp == "nhwc":
#             if out4.shape[-1] == 3:
#                 return out4[0]
#             if out4.shape[1] == 3:
#                 return np.transpose(out4[0], (1,2,0))
#         elif exp == "nchw":
#             if out4.shape[1] == 3:
#                 return np.transpose(out4[0], (1,2,0))
#             if out4.shape[-1] == 3:
#                 return out4[0]

#         if out4.shape[-1] == 3:
#             return out4[0]
#         if out4.shape[1] == 3:
#             return np.transpose(out4[0], (1,2,0))
#         raise RuntimeError(f"Cannot interpret output layout: {out4.shape}")

#     pred_hwc = out_to_hwc(out_u8)
#     return pred_hwc, (Hm, Wm), (y0, x0), (pad_h, pad_w), layout_in

# def map_lr_to_hr_name(lr_name: str, scale: int) -> str:
#     stem, ext = os.path.splitext(lr_name)
#     suf = f"x{scale}"
#     if stem.endswith(suf):
#         stem = stem[:-len(suf)]
#     return stem + ext

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--tflite", required=True)
#     ap.add_argument("--lr_dir", required=True)
#     ap.add_argument("--hr_dir", required=True)
#     ap.add_argument("--glob", default="*.png")
#     ap.add_argument("--num", type=int, default=0)
#     ap.add_argument("--threads", type=int, default=4)
#     ap.add_argument("--warmup", type=int, default=1)
#     ap.add_argument("--crop", type=int, default=0)
#     ap.add_argument("--mode", choices=["rgb", "y"], default="rgb")
#     ap.add_argument("--expect_out", choices=["auto","nhwc","nchw"], default="auto")
#     ap.add_argument("--scale", type=int, default=3)
#     args = ap.parse_args()

#     lr_files = sorted(glob.glob(os.path.join(args.lr_dir, args.glob)))
#     if not lr_files:
#         raise SystemExit("No LR files found.")
#     if args.num and args.num > 0:
#         lr_files = lr_files[:args.num]

#     itp = tflite.Interpreter(model_path=args.tflite, num_threads=args.threads)
#     itp.allocate_tensors()
#     in_det = itp.get_input_details()[0]
#     out_det = itp.get_output_details()[0]

#     # Warmup
#     lr0 = load_img(lr_files[0])
#     for _ in range(max(args.warmup, 0)):
#         _ = infer_one(itp, lr0, in_det, out_det, expect=args.expect_out)

#     scores = []
#     t0 = time.time()

#     for p in lr_files:
#         lr_name = os.path.basename(p)
#         hr_name = map_lr_to_hr_name(lr_name, args.scale)
#         gt_path = os.path.join(args.hr_dir, hr_name)
#         if not os.path.exists(gt_path):
#             raise SystemExit(f"Missing GT: {gt_path}")

#         lr = load_img(p)
#         gt = load_img(gt_path)

#         pred, (Hm, Wm), (y0, x0), (pad_h, pad_w), _layout_in = infer_one(
#             itp, lr, in_det, out_det, expect=args.expect_out
#         )

#         # If LR was center-cropped, crop GT accordingly (alignment fix)
#         if y0 != 0 or x0 != 0:
#             gy0 = y0 * args.scale
#             gx0 = x0 * args.scale
#             gt = gt[gy0:gy0 + pred.shape[0], gx0:gx0 + pred.shape[1], :]

#         # Remove SR region corresponding to padded LR bottom/right
#         if pad_h > 0 or pad_w > 0:
#             H_keep = pred.shape[0] - pad_h * args.scale
#             W_keep = pred.shape[1] - pad_w * args.scale
#             pred = pred[:H_keep, :W_keep, :]

#         # Crop to GT size (GT authoritative after alignment)
#         pred = pred[:gt.shape[0], :gt.shape[1], :]

#         pred_c = crop_border(pred, args.crop)
#         gt_c   = crop_border(gt, args.crop)

#         if pred_c.shape != gt_c.shape:
#             raise SystemExit(f"Shape mismatch: pred {pred_c.shape} vs gt {gt_c.shape} for {lr_name}")

#         if args.mode == "rgb":
#             s = psnr(pred_c, gt_c)
#         else:
#             s = psnr(rgb2y_bt601(pred_c), rgb2y_bt601(gt_c))
#         scores.append(s)

#     dt = time.time() - t0
#     print(f"Images: {len(scores)} | PSNR({args.mode}, crop={args.crop}): {sum(scores)/len(scores):.4f} dB | time={dt:.2f}s")

# if __name__ == "__main__":
#     main()






#!/usr/bin/env python3
import argparse, glob, random, math
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# ---------- helpers ----------
def load_rgb_255(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)  # HWC [0..255]

def rgb_to_y(x: np.ndarray) -> np.ndarray:
    # x: HWC float
    r, g, b = x[..., 0:1], x[..., 1:2], x[..., 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b

def shave_border(a: np.ndarray, shave: int) -> np.ndarray:
    if shave <= 0:
        return a
    h, w = a.shape[0], a.shape[1]
    if h <= 2 * shave or w <= 2 * shave:
        return a
    return a[shave:-shave, shave:-shave, :]

def psnr_255(a: np.ndarray, b: np.ndarray, eps=1e-12) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse < eps:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)

def gaussian_kernel_11x11(sigma=1.5) -> np.ndarray:
    ax = np.arange(-5, 6)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k /= np.sum(k)
    return k.astype(np.float32)

def conv2d_valid(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    # img: HxW float32, k: 11x11
    H, W = img.shape
    kh, kw = k.shape
    out_h, out_w = H - kh + 1, W - kw + 1
    out = np.zeros((out_h, out_w), dtype=np.float32)
    # naive conv (ok cho eval vài chục ảnh)
    for i in range(out_h):
        patch = img[i:i+kh, :]
        for j in range(out_w):
            out[i, j] = float(np.sum(patch[:, j:j+kw] * k))
    return out

def ssim_single_channel(x: np.ndarray, y: np.ndarray) -> float:
    # x,y: HxW float32 range [0..255]
    k = gaussian_kernel_11x11(1.5)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_x = conv2d_valid(x, k)
    mu_y = conv2d_valid(y, k)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = conv2d_valid(x * x, k) - mu_x2
    sigma_y2 = conv2d_valid(y * y, k) - mu_y2
    sigma_xy = conv2d_valid(x * y, k) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + 1e-12)
    return float(np.mean(ssim_map))

def ssim_rgb(x: np.ndarray, y: np.ndarray) -> float:
    # mean SSIM over channels
    vals = []
    for c in range(3):
        vals.append(ssim_single_channel(x[..., c], y[..., c]))
    return float(np.mean(vals))

def quantize_input(x: np.ndarray, dtype, qinfo):
    # qinfo: (scale, zero_point) or list; handle per-tensor
    scale, zp = qinfo
    if scale == 0:
        return x.astype(dtype)
    q = np.round(x / scale + zp)
    if dtype == np.int8:
        q = np.clip(q, -128, 127)
    elif dtype == np.uint8:
        q = np.clip(q, 0, 255)
    return q.astype(dtype)

def dequant_output(xq: np.ndarray, qinfo):
    scale, zp = qinfo
    if scale == 0:
        return xq.astype(np.float32)
    return (xq.astype(np.float32) - float(zp)) * float(scale)

def crop_pair(lr: np.ndarray, hr: np.ndarray, H: int, W: int, scale: int = 3):
    # lr: Hlr x Wlr x 3 ; hr: Hhr x Whr x 3
    h, w = lr.shape[:2]

    # nếu ảnh LR nhỏ -> resize LR lên (W,H) và HR lên (W*3,H*3)
    if h < H or w < W:
        lr_img = Image.fromarray(np.clip(lr,0,255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
        hr_img = Image.fromarray(np.clip(hr,0,255).astype(np.uint8)).resize((W*scale, H*scale), Image.BICUBIC)
        return np.array(lr_img, np.float32), np.array(hr_img, np.float32)

    y = 0 if h == H else random.randint(0, h - H)
    x = 0 if w == W else random.randint(0, w - W)
    lr_c = lr[y:y+H, x:x+W, :]

    hr_y, hr_x = y * scale, x * scale
    hr_c = hr[hr_y:hr_y + H*scale, hr_x:hr_x + W*scale, :]
    return lr_c.astype(np.float32), hr_c.astype(np.float32)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lr_dir", required=True)  # DIV2K_valid_LR_bicubic/X3
    ap.add_argument("--hr_dir", required=True)  # DIV2K_valid_HR
    ap.add_argument("--num_images", type=int, default=20)
    ap.add_argument("--crops_per_image", type=int, default=1)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    itp = tflite.Interpreter(model_path=args.model, num_threads=4)
    itp.allocate_tensors()
    in0 = itp.get_input_details()[0]
    out0 = itp.get_output_details()[0]

    in_dtype = in0["dtype"]
    out_dtype = out0["dtype"]

    in_q = in0.get("quantization", (0.0, 0))
    out_q = out0.get("quantization", (0.0, 0))

    # input/output names/index
    in_index = in0["index"]
    out_index = out0["index"]

    # file list (match by stem)
    hr_paths = sorted(glob.glob(args.hr_dir.rstrip("/") + "/*.png"))
    if not hr_paths:
        raise FileNotFoundError("No HR pngs found")
    random.shuffle(hr_paths)
    hr_paths = hr_paths[: max(1, args.num_images)]

    # metrics accum
    shaves = [0, 3]
    acc = {f"psnr_rgb_sh{s}": [] for s in shaves}
    acc.update({f"psnr_y_sh{s}": [] for s in shaves})
    acc.update({f"ssim_rgb_sh{s}": [] for s in shaves})
    acc.update({f"ssim_y_sh{s}": [] for s in shaves})

    for hr_p in hr_paths:
        stem = hr_p.split("/")[-1].replace(".png", "")
        lr_p = f"{args.lr_dir.rstrip('/')}/{stem}x{args.scale}.png"
        if not glob.glob(lr_p):
            # skip if missing pair
            continue

        hr = load_rgb_255(hr_p)
        lr = load_rgb_255(lr_p)

        for _ in range(max(1, args.crops_per_image)):
            lr_c, hr_c = crop_pair(lr, hr, args.h, args.w, scale=args.scale)

            # prepare input NHWC
            inp = lr_c[None, ...]  # 1,H,W,3 float32 [0..255]
            if in_dtype != np.float32:
                inp_q = quantize_input(inp, in_dtype, in_q)
            else:
                inp_q = inp.astype(np.float32)

            itp.set_tensor(in_index, inp_q)
            itp.invoke()
            out = itp.get_tensor(out_index)

            # dequant output if needed
            if out_dtype != np.float32:
                sr = dequant_output(out, out_q)
            else:
                sr = out.astype(np.float32)

            # sr expected NHWC
            sr = sr[0]
            # align
            Hh, Wh = hr_c.shape[0], hr_c.shape[1]
            sr = sr[:Hh, :Wh, :]

            # clamp
            sr = np.clip(sr, 0.0, 255.0)
            gt = np.clip(hr_c, 0.0, 255.0)

            for sh in shaves:
                sr_s = shave_border(sr, sh)
                gt_s = shave_border(gt, sh)

                acc[f"psnr_rgb_sh{sh}"].append(psnr_255(sr_s, gt_s))
                acc[f"ssim_rgb_sh{sh}"].append(ssim_rgb(sr_s, gt_s))

                sr_y = rgb_to_y(sr_s)[..., 0]
                gt_y = rgb_to_y(gt_s)[..., 0]
                acc[f"psnr_y_sh{sh}"].append(psnr_255(sr_y[..., None], gt_y[..., None]))
                acc[f"ssim_y_sh{sh}"].append(ssim_single_channel(sr_y, gt_y))

    # report
    print("Model:", args.model)
    print("Input dtype:", in_dtype, "quant:", in_q)
    print("Output dtype:", out_dtype, "quant:", out_q)
    for k, v in acc.items():
        if len(v):
            print(f"{k}: {float(np.mean(v)):.4f} (n={len(v)})")
        else:
            print(f"{k}: n=0")

if __name__ == "__main__":
    main()
