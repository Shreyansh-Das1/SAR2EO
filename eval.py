import os
import argparse
import glob
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def parse_args():
    parser = argparse.ArgumentParser(description="GalaxEye Official Cross-Modal Evaluation Metric Suite")
    # Mandated CLI keys from the official project contract
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing the model's generated RGB optical images")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing the target ground-truth optical images")
    return parser.parse_args()

def calculate_metrics():
    args = parse_args()
    
    # Locate all prediction items
    pred_path_pattern = os.path.join(args.pred_dir, "*.png")
    pred_files = glob.glob(pred_path_pattern)
    
    if not pred_files:
        raise FileNotFoundError(f"No .png predictions found inside: {args.pred_dir}")
        
    psnr_scores = []
    ssim_scores = []
    mae_scores = []
    processed_count = 0

    print(f"🔬 Auditing evaluation cross-match for {len(pred_files)} files...")

    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        gt_path = os.path.join(args.gt_dir, filename)
        
        # Check if corresponding ground truth exists
        if not os.path.exists(gt_path):
            print(f"Warning: Missing ground-truth match for {filename}. Skipping patch.")
            continue
            
        # Load images cleanly as RGB matrices [0, 255]
        img_pred = np.array(Image.open(pred_path).convert('RGB'))
        img_gt = np.array(Image.open(gt_path).convert('RGB'))
        
        # 1. Structural Similarity Index (SSIM) - Explicit channel axis tracking
        score_ssim = ssim(img_gt, img_pred, channel_axis=2, data_range=255)
        
        # 2. Peak Signal-to-Noise Ratio (PSNR)
        score_psnr = psnr(img_gt, img_pred, data_range=255)
        
        # 3. Mean Absolute Error (MAE)
        score_mae = np.mean(np.abs(img_gt.astype(np.float32) - img_pred.astype(np.float32)))
        
        psnr_scores.append(score_psnr)
        ssim_scores.append(score_ssim)
        mae_scores.append(score_mae)
        processed_count += 1

    if processed_count == 0:
        print("Error: Zero matched filename pairs found between predictions and ground truth.")
        return

    # Aggregate statistics
    mean_psnr = np.mean(psnr_scores)
    mean_ssim = np.mean(ssim_scores)
    mean_mae = np.mean(mae_scores)

    print("\n" + "="*40)
    print("OFFICIAL VALIDATION METRICS REPORT")
    print("="*40)
    print(f"Processed Matches : {processed_count} frames")
    print(f"Mean PSNR         : {mean_psnr:.4f} dB")
    print(f"Mean SSIM         : {mean_ssim:.4f}")
    print(f"Mean MAE (L1)     : {mean_mae:.4f} pixels")
    print("="*40 + "\n")

if __name__ == "__main__":
    calculate_metrics()