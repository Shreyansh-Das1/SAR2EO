import os
import sys
import yaml
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add root folder to paths to ensure safe relative cross-imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import SODataset
from models import UNetGenerator

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_visualizer():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate the Leak-Proof Test Dataset Split
    test_dataset = SODataset(base_dir=cfg['dataset_path'], split='test', seed=cfg['seed'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 2. Instantiate and Arm the Two Ablation Generators
    net_G_A = UNetGenerator(input_nc=1, output_nc=3).to(device)
    net_G_B = UNetGenerator(input_nc=1, output_nc=3).to(device)
    
    checkpoint_path_A = os.path.join("/kaggle/working/checkpoints_ablation_A", "generator_epoch_30.pth")
    checkpoint_path_B = os.path.join("/kaggle/working/checkpoints", "generator_epoch_40.pth")
    
    if not os.path.exists(checkpoint_path_A) or not os.path.exists(checkpoint_path_B):
        print("⚠️ Warning: One or both best checkpoints not resolved at target locations.")
        print(f"Looking for A at: {checkpoint_path_A}")
        print(f"Looking for B at: {checkpoint_path_B}")
        print("Please verify your checkpoint path names before running this plot engine.")
        return

    net_G_A.load_state_dict(torch.load(checkpoint_path_A, map_location=device))
    net_G_B.load_state_dict(torch.load(checkpoint_path_B, map_location=device))
    
    net_G_A.eval()
    net_G_B.eval()
    
    scenarios = [
        "Success 1:\nUrban Infrastructure",
        "Success 2:\nAgricultural Fields",
        "Success 3:\nCoastal Boundary",
        "Failure 1:\nTopographic Shadow",
        "Failure 2:\nSoil/Forest Ambiguity"
    ]
    
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(16, 18))
    
    # Fixed Title configuration using absolute coordinates to ensure zero overlapping boundaries
    fig.text(0.55, 0.96, "Qualitative Model Comparison: Ablation A vs. Proposed Ablation B", 
             fontsize=16, weight='bold', ha='center')
    
    col_titles = [
        "SAR Input\n(Sentinel-1 Grayscale)", 
        "Ablation A Output\n(Pure L1 Baseline)", 
        "Ablation B Output\n(Proposed cGAN Model)", 
        "Ground Truth\n(Sentinel-2 RGB Optical)"
    ]
    
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=11, pad=14, weight='bold', va='baseline')
        
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            if count >= 5:
                break
                
            real_sar = batch['s1'].to(device)
            real_eo = batch['s2'].to(device)
            
            # Forward passes through both domain engines
            fake_eo_A = net_G_A(real_sar)
            fake_eo_B = net_G_B(real_sar)
            
            # Denormalize tensors from operational range [-1, 1] back to standard image range [0, 1]
            sar_img = (real_sar.squeeze().cpu().numpy() + 1.0) / 2.0
            eo_A_img = (fake_eo_A.squeeze().cpu().permute(1, 2, 0).clamp(-1.0, 1.0).numpy() + 1.0) / 2.0
            eo_B_img = (fake_eo_B.squeeze().cpu().permute(1, 2, 0).clamp(-1.0, 1.0).numpy() + 1.0) / 2.0
            gt_img = (real_eo.squeeze().cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0
            
            # Row Layout Generation
            axes[count, 0].imshow(sar_img, cmap='gray')
            axes[count, 1].imshow(eo_A_img)
            axes[count, 2].imshow(eo_B_img)
            axes[count, 3].imshow(gt_img)
            
            # Clean tick markers and add descriptive scenario annotations
            for col_idx in range(4):
                axes[count, col_idx].set_xticks([])
                axes[count, col_idx].set_yticks([])
            
            # Decoupled absolute coordinate text injection to avoid y-label overlapping anomalies
            axes[count, 0].text(-0.12, 0.5, scenarios[count], 
                                transform=axes[count, 0].transAxes, 
                                fontsize=11, weight='bold', 
                                ha='right', va='center', 
                                rotation=0)
            count += 1

    # Save out the compiled result panel grid
    out_dir = "/kaggle/working/outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ablation_comparison_grid.png")
    
    # tight_layout handles internal subplot edges; subplots_adjust isolates outer margins
    plt.tight_layout()
    fig.subplots_adjust(top=0.91, left=0.20, hspace=0.18, wspace=0.06)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Production evaluation sheet successfully saved at: {out_path}")

if __name__ == "__main__":
    run_visualizer()
