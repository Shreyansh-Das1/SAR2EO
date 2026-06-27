import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class SODataset(Dataset):
    def __init__(self, base_dir, split='train', split_ratio=(0.8, 0.1, 0.1), seed=42):
        self.base_dir = base_dir
        self.split = split
        
        all_terrain_dirs = sorted([
            d for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d))
        ])
        
        random.seed(seed)
        random.shuffle(all_terrain_dirs)
        
        num_folders = len(all_terrain_dirs)
        
        # Robust fallback for small folder sets (e.g., exactly 4 folders)
        if num_folders <= 4:
            if split == 'train':
                self.selected_folders = all_terrain_dirs[:2]   # 2 folders (50%)
            elif split == 'val':
                self.selected_folders = all_terrain_dirs[2:3]  # 1 folder (25%)
            elif split == 'test':
                self.selected_folders = all_terrain_dirs[3:]   # 1 folder (25%)
        else:
            # Standard ratio split for large datasets
            train_end = int(split_ratio[0] * num_folders)
            val_end = train_end + int(split_ratio[1] * num_folders)
            
            if split == 'train':
                self.selected_folders = all_terrain_dirs[:train_end]
            elif split == 'val':
                self.selected_folders = all_terrain_dirs[train_end:val_end]
            elif split == 'test':
                self.selected_folders = all_terrain_dirs[val_end:]
                
        if not self.selected_folders:
            raise ValueError(f"Split '{split}' received 0 folders. Check split ratios or directory structure.")
            
        self.pairs = []
        for folder in self.selected_folders:
            sar_dir = os.path.join(base_dir, folder, 's1')  
            eo_dir = os.path.join(base_dir, folder, 's2')
            
            sar_images = sorted(glob.glob(os.path.join(sar_dir, '*.png')))
            for sar_path in sar_images:
                filename = os.path.basename(sar_path)
                eo_filename = filename.replace('_s1_', '_s2_')
                eo_path = os.path.join(eo_dir, eo_filename)
                
                if os.path.exists(eo_path):
                    self.pairs.append((sar_path, eo_path))

    def __len__(self):
        return len(self.pairs)

    def preprocess_sar(self, sar_img):
        sar_arr = np.array(sar_img, dtype=np.float32)
        sar_normalized = (sar_arr / 127.5) - 1.0
        sar_tensor = torch.tensor(sar_normalized).unsqueeze(0)
        return sar_tensor

    def preprocess_eo(self, eo_img):
        eo_arr = np.array(eo_img, dtype=np.float32)
        eo_normalized = (eo_arr / 127.5) - 1.0
        eo_tensor = torch.tensor(eo_normalized).permute(2, 0, 1)
        return eo_tensor

    def __getitem__(self, idx):
        sar_path, eo_path = self.pairs[idx]
        
        sar_img = Image.open(sar_path).convert('L')  
        eo_img = Image.open(eo_path).convert('RGB') 
        
        sar_tensor = self.preprocess_sar(sar_img)
        eo_tensor = self.preprocess_eo(eo_img)
        
        return {
            's1': sar_tensor, 
            's2': eo_tensor,
            'filename': os.path.basename(sar_path)
        }
