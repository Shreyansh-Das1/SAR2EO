import os
import yaml
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import custom structures developed in previous phases
from dataset import SODataset
from models import UNetGenerator, PatchGANDiscriminator

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_trainer():
    # 1. Initialize Configuration and Directory Trees
    cfg = load_config()
    torch.manual_seed(cfg['seed'])
    
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['output_dir'], exist_ok=True)
    
    # Track the total start time of the entire script execution
    start_wall_clock = time.time()
    
    # 2. Build Structural Data Loaders
    train_dataset = SODataset(base_dir=cfg['dataset_path'], split='train', seed=cfg['seed'])
    val_dataset = SODataset(base_dir=cfg['dataset_path'], split='val', seed=cfg['seed'])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)
    
    # 3. Instantiate Networks & Shift to Target Compute Accelerator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_G = UNetGenerator(input_nc=1, output_nc=3).to(device)
    
    # Define Basic Loss Criteria
    criterion_L1 = nn.L1Loss()
    
    # Configure Structural Ablation Environment Boundaries
    mode = cfg['ablation_mode']
    if mode == "B":
        net_D = PatchGANDiscriminator(input_nc=4).to(device) # Concatenated conditional pairs (1+3=4 channels)
        criterion_GAN = nn.BCEWithLogitsLoss()
        
        optimizer_G = optim.Adam(net_G.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))
        optimizer_D = optim.Adam(net_D.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))
    else:
        optimizer_G = optim.Adam(net_G.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))
        net_D, criterion_GAN, optimizer_D = None, None, None

    # Open Log Records File to Track Loss Distributions Globally (with temporal tracking columns)
    history_file = os.path.join(cfg['output_dir'], "loss_history.csv")
    with open(history_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        if mode == "B":
            writer.writerow(["epoch", "train_G", "train_D", "val_L1", "epoch_duration_sec", "cumulative_time_min"])
        else:
            writer.writerow(["epoch", "train_G", "val_L1", "epoch_duration_sec", "cumulative_time_min"])

    print(f"Starting Engine Execution Sequence under Ablation Framework: [Mode {mode}]")
    
    # 4. Main Model Optimization Execution Epochs
    for epoch in range(1, cfg['epochs'] + 1):
        epoch_start_time = time.time() # Start stopwatch for this specific epoch
        
        net_G.train()
        if net_D: net_D.train()
        
        running_g_loss = 0.0
        running_d_loss = 0.0
        
        for batch in train_loader:
            real_sar = batch['s1'].to(device)
            real_eo = batch['s2'].to(device)
            
            # --- Optimization Mode B: Adversarial Execution Flow ---
            if mode == "B":
                # Update Discriminator Graph Boundaries
                optimizer_D.zero_grad()
                fake_eo = net_G(real_sar)
                
                # Evaluation Pass 1: True Ground-Truth Distribution
                real_pair = torch.cat((real_sar, real_eo), dim=1)
                pred_real = net_D(real_pair)
                loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
                
                # Evaluation Pass 2: Synthesized Surface Distribution
                fake_pair = torch.cat((real_sar, fake_eo.detach()), dim=1)
                pred_fake = net_D(fake_pair)
                loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                optimizer_D.step()
                running_d_loss += loss_D.item()
                
                # Update Generator Graph Boundaries
                optimizer_G.zero_grad()
                fake_pair_G = torch.cat((real_sar, fake_eo), dim=1)
                pred_fake_G = net_D(fake_pair_G)
                
                loss_G_GAN = criterion_GAN(pred_fake_G, torch.ones_like(pred_fake_G))
                loss_G_L1 = criterion_L1(fake_eo, real_eo) * cfg['lambda_l1']
                
                loss_G = loss_G_GAN + loss_G_L1
                loss_G.backward()
                optimizer_G.step()
                running_g_loss += loss_G.item()
                
            # --- Optimization Mode A: Pure Linear L1 Baseline Flow ---
            else:
                optimizer_G.zero_grad()
                fake_eo = net_G(real_sar)
                loss_G = criterion_L1(fake_eo, real_eo)
                loss_G.backward()
                optimizer_G.step()
                running_g_loss += loss_G.item()

        # 5. Validation Tracking Pass
        net_G.eval()
        val_l1_accum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                real_sar = batch['s1'].to(device)
                real_eo = batch['s2'].to(device)
                fake_eo = net_G(real_sar)
                val_l1_accum += criterion_L1(fake_eo, real_eo).item()
                
        epoch_g = running_g_loss / len(train_loader)
        epoch_v = val_l1_accum / len(val_loader)
        
        # Calculate temporal metrics
        epoch_end_time = time.time()
        duration_seconds = epoch_end_time - epoch_start_time
        cumulative_minutes = (epoch_end_time - start_wall_clock) / 60.0
        
        # Write metrics and timing statistics directly out to CSV file
        with open(history_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if mode == "B":
                epoch_d = running_d_loss / len(train_loader)
                writer.writerow([epoch, epoch_g, epoch_d, epoch_v, f"{duration_seconds:.2f}", f"{cumulative_minutes:.2f}"])
                print(f"Epoch [{epoch}/{cfg['epochs']}] -> G_Loss: {epoch_g:.4f} | D_Loss: {epoch_d:.4f} | Val_L1: {epoch_v:.4f} | Time: {duration_seconds:.1f}s | Total: {cumulative_minutes:.1f}m")
            else:
                writer.writerow([epoch, epoch_g, epoch_v, f"{duration_seconds:.2f}", f"{cumulative_minutes:.2f}"])
                print(f"Epoch [{epoch}/{cfg['epochs']}] -> G_Loss: {epoch_g:.4f} | Val_L1: {epoch_v:.4f} | Time: {duration_seconds:.1f}s | Total: {cumulative_minutes:.1f}m")

        # Persist Best Checkpoints Locally
        if epoch % 10 == 0 or epoch == cfg['epochs']:
            torch.save(net_G.state_dict(), os.path.join(cfg['checkpoint_dir'], f"generator_epoch_{epoch}.pth"))

if __name__ == "__main__":
    run_trainer()
