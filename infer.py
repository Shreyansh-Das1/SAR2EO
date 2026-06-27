import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image

# Force paths to look inside your submission module root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import UNetGenerator

def parse_args():
    parser = argparse.ArgumentParser(description="GalaxEye Official Cross-Modal Inference Contract Test-Bed")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the single-channel input SAR image (.png)")
    parser.add_argument("--output_path", type=str, required=True, help="Destination file path to save the generated RGB optical output")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to your best trained weights file (.pth)")
    return parser.parse_args()

@torch.no_grad()
def execute_inference():
    args = parse_args()
    
    # 1. Establish offline environment computing accelerator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Instantiate and compile model weight maps
    model = UNetGenerator(input_nc=1, output_nc=3)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"❌ Target weights matrix not found at: {args.checkpoint}")
        
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Apply Preprocessing Pipeline
    # Load input image as single-channel grayscale ('L')
    raw_sar = Image.open(args.input_path).convert('L')
    sar_arr = np.array(raw_sar, dtype=np.float32)
    
    # Lineally normalize standard pixel bounds to structural domain range [-1, 1]
    sar_normalized = (sar_arr / 127.5) - 1.0
    
    # Expand to tensor dimensions: (Batch=1, Channels=1, Height, Width)
    sar_tensor = torch.tensor(sar_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # 4. Forward Inference Pass through the Generator Network
    fake_eo_tensor = model(sar_tensor)
    
    # 5. Denormalization Pipeline back to 8-bit Image Output Space
    # Strip batch dimension and map back from [-1, 1] to standard RGB [0, 255]
    fake_eo = fake_eo_tensor.squeeze(0).cpu().clamp(-1.0, 1.0)
    fake_eo_arr = fake_eo.permute(1, 2, 0).numpy()  # Reshape to (H, W, C)
    fake_eo_denormalized = ((fake_eo_arr + 1.0) * 127.5).astype(np.uint8)
    
    # 6. Save image to disk
    out_img = Image.open(args.input_path).convert('RGB') # Baseline template match shape validation
    out_img = Image.fromarray(fake_eo_denormalized)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    out_img.save(args.output_path)
    print(f"✅ Production Inference Matrix successfully stored at: {args.output_path}")

if __name__ == "__main__":
    execute_inference()
