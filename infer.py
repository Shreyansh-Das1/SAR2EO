import os
import sys
import argparse
import glob
import numpy as np
import torch
from PIL import Image

# Force paths to look inside your submission module root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import UNetGenerator

def parse_args():
    parser = argparse.ArgumentParser(description="GalaxEye Official Cross-Modal Inference Contract Test-Bed")
    # Exact CLI flags mandated by the contract
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing single-channel SAR patches (.png)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated RGB optical outputs")
    parser.add_argument("--weights", type=str, required=True, help="Path to your best trained weights file (.pth)")
    return parser.parse_args()

@torch.no_grad()
def execute_inference():
    args = parse_args()
    
    # 1. Establish offline environment computing accelerator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Instantiate and compile model weight maps
    model = UNetGenerator(input_nc=1, output_nc=3)
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Target weights matrix not found at: {args.weights}")
        
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3. Find all PNG images in the input directory
    search_path = os.path.join(args.input_dir, "*.png")
    input_files = glob.glob(search_path)
    
    if not input_files:
        print(f"No .png files found in input directory: {args.input_dir}")
        return

    print(f"Found {len(input_files)} patches to process...")

    for img_path in input_files:
        filename = os.path.basename(img_path)
        output_path = os.path.join(args.output_dir, filename)
        
        # Apply Preprocessing Pipeline
        raw_sar = Image.open(img_path).convert('L')
        sar_arr = np.array(raw_sar, dtype=np.float32)
        
        # Lineally normalize standard pixel bounds to structural domain range [-1, 1]
        sar_normalized = (sar_arr / 127.5) - 1.0
        
        # Expand to tensor dimensions: (Batch=1, Channels=1, Height, Width)
        sar_tensor = torch.tensor(sar_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # 4. Forward Inference Pass through the Generator Network
        fake_eo_tensor = model(sar_tensor)
        
        # 5. Denormalization Pipeline back to 8-bit Image Output Space
        fake_eo = fake_eo_tensor.squeeze(0).cpu().clamp(-1.0, 1.0)
        fake_eo_arr = fake_eo.permute(1, 2, 0).numpy()  # Reshape to (H, W, C)
        fake_eo_denormalized = ((fake_eo_arr + 1.0) * 127.5).astype(np.uint8)
        
        # 6. Save image to disk preserving the filename
        out_img = Image.fromarray(fake_eo_denormalized)
        out_img.save(output_path)
        
    print(f"Production Inference Matrix successfully stored at: {args.output_dir}")

if __name__ == "__main__":
    execute_inference()
