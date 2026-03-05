import argparse
import os
import glob
import time
import numpy as np
from PIL import Image
import tensorflow as tf

# Try to import skimage for SSIM (standard in SR evaluation)
try:
    from skimage.metrics import structural_similarity as ssim_metric
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
except ImportError:
    print("Please install scikit-image: pip install scikit-image")
    exit(1)

def load_image(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)

def calculate_metrics(pred, hr, scale=3):
    """
    Calculates PSNR and SSIM. 
    Standard SR competitions "shave" the boundaries by the scale factor 
    to ignore edge artifacts caused by padding.
    """
    # Shave boundaries
    if scale > 0:
        pred = pred[scale:-scale, scale:-scale, :]
        hr = hr[scale:-scale, scale:-scale, :]
        
    # Clip to valid range and round to nearest integer (8-bit color depth)
    pred = np.clip(np.round(pred), 0, 255).astype(np.uint8)
    hr = np.clip(np.round(hr), 0, 255).astype(np.uint8)
    
    psnr = psnr_metric(hr, pred, data_range=255)
    ssim = ssim_metric(hr, pred, channel_axis=2, data_range=255)
    
    return psnr, ssim

def evaluate_tflite_model(tflite_path, lr_paths, hr_paths, scale=3, num_warmup=5):
    """Runs full evaluation on a single TFLite model."""
    if not os.path.exists(tflite_path):
        return None, None, None
        
    print(f"Loading {os.path.basename(tflite_path)}...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path, num_threads=4)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    psnr_list, ssim_list, latency_list = [], [], []
    
    for idx, (lr_p, hr_p) in enumerate(zip(lr_paths, hr_paths)):
        lr_img = load_image(lr_p)
        hr_img = load_image(hr_p)
        
        # Prepare Input (Assuming NHWC layout from litert_torch export)
        # Add batch dimension
        input_data = np.expand_dims(lr_img, axis=0)
        
        # Resize interpreter if input shape differs from exported shape
        if list(input_details['shape']) != list(input_data.shape):
            interpreter.resize_tensor_input(input_details['index'], input_data.shape)
            interpreter.allocate_tensors()
            
        interpreter.set_tensor(input_details['index'], input_data)
        
        # Warmup for latency stability (only on first image)
        if idx == 0:
            for _ in range(num_warmup):
                interpreter.invoke()
                
        # Inference & Time Measurement
        start_time = time.perf_counter()
        interpreter.invoke()
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000.0
        latency_list.append(latency_ms)
        
        # Get Output
        pred = interpreter.get_tensor(output_details['index'])[0] # Remove batch dim
        
        # Calculate Metrics
        psnr, ssim = calculate_metrics(pred, hr_img, scale=scale)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_latency = np.mean(latency_list)
    
    return avg_psnr, avg_ssim, avg_latency

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_dir", type=str, required=True, help="Path to DIV2K Valid LR (x3)")
    parser.add_argument("--hr_dir", type=str, required=True, help="Path to DIV2K Valid HR")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--num_images", type=int, default=50, help="How many images to test")
    args = parser.parse_args()

    # Discover images
    lr_paths = sorted(glob.glob(os.path.join(args.lr_dir, "*.png")))[:args.num_images]
    hr_paths = sorted(glob.glob(os.path.join(args.hr_dir, "*.png")))[:args.num_images]
    
    assert len(lr_paths) > 0 and len(lr_paths) == len(hr_paths), "Dataset paths invalid or mismatched."

    # Define the models we want to compare
    models_to_test = {
        "Baseline (PTQ)": "antsr_int8.tflite",
        "Idea 1 (RepQ-SR)": "antsr_repq_int8.tflite",
        "Idea 2 (F-QKD)": "antsr_fqkd_int8.tflite",
        "Idea 3 (PACT)": "antsr_pact_int8.tflite",
        # "Idea 1+3 (Best)": "antsr_repq_pact_int8.tflite" # If you combine them!
    }

    results = {}
    
    print(f"\nEvaluating on {len(lr_paths)} images...")
    print("-" * 75)
    print(f"{'Model Name':<20} | {'PSNR (dB)':<12} | {'SSIM':<10} | {'CPU Latency (ms)':<15}")
    print("-" * 75)

    for model_name, path in models_to_test.items():
        if not os.path.exists(path):
            print(f"{model_name:<20} | {'FILE NOT FOUND':<12} | {'-':<10} | {'-':<15}")
            continue
            
        psnr, ssim, latency = evaluate_tflite_model(path, lr_paths, hr_paths, args.scale)
        
        if psnr is not None:
            results[model_name] = {"psnr": psnr, "ssim": ssim, "latency": latency}
            print(f"{model_name:<20} | {psnr:>10.4f}   | {ssim:>8.4f} | {latency:>10.2f} ms")

    print("-" * 75)
    
    # Identify the winner based on PSNR
    if results:
        winner = max(results, key=lambda k: results[k]['psnr'])
        print(f"\n🏆 WINNER (Highest PSNR): {winner}")
        print("Note: In MAI, all INT8 variants will have identical edge latency on the target NPU.")
        print("Submit the model with the highest PSNR to CodaLab.")

if __name__ == "__main__":
    main()