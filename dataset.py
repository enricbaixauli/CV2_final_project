import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch
import scipy.io as sio
import utils
import cv2    
import face_alignment
import numpy as np
import scipy.io
import random

class I2HeadDataset(Dataset):
    def __init__(self, image_paths_or_dir, transform=None, occlusion_mode="raw"):
        """
        image_paths_or_dir: either a list of raw image paths or path to the 'raw' directory.
        transform: torchvision transforms pipeline.
        occlusion_mode: 'raw', 'mask', 'glasses', 'both', or 'random' (for balanced training).
        """
        self.transform = transform
        self.occlusion_mode = occlusion_mode

        if isinstance(image_paths_or_dir, str) and os.path.isdir(image_paths_or_dir):
            dataset_dir = image_paths_or_dir
            self.image_paths = [
                os.path.join(dataset_dir, f)
                for f in sorted(os.listdir(dataset_dir))
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
        else:
            self.image_paths = list(image_paths_or_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        raw_img_path = self.image_paths[idx]

        # We have 4 branches to coose from
        mode = self.occlusion_mode
        if mode == "random":
            mode = random.choice(["raw", "mask", "glasses", "both"])

        # Based on them we determine the path
        if mode == "raw":
            img_path = raw_img_path
        else:
            # Move from 'dataset/test/raw/filename.png' to 'dataset/test/mode/filename_mode.png'
            raw_dir, filename = os.path.split(raw_img_path)
            parent_dir = os.path.dirname(raw_dir) # up to 'dataset/test'
            basename, ext = os.path.splitext(filename)
            
            if basename.endswith("_raw"):
                new_basename = basename[:-4] + f"_{mode}"
            else:
                new_basename = f"{basename}_{mode}"
                
            img_path = os.path.join(parent_dir, mode, new_basename + ext)

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Load the corresponding .mat file directly from the same folder
        base_path, _ = os.path.splitext(img_path)
        mat_path = base_path + ".mat"
            
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Missing .mat file for image: {mat_path}")

        mat_data = scipy.io.loadmat(mat_path)
        vector_6d = mat_data['HP_camera'][0]

        # Extract angles (originally in degrees)
        angle_1 = vector_6d[3]
        angle_2 = vector_6d[4]
        angle_3 = vector_6d[5]

        # Convert degrees to radians
        rad_1 = angle_1 * (np.pi / 180.0)
        rad_2 = angle_2 * (np.pi / 180.0)
        rad_3 = angle_3 * (np.pi / 180.0)

        # Align order
        pitch, yaw, roll = rad_1, rad_2, rad_3

        poses = torch.tensor([pitch, yaw, roll], dtype=torch.float32)

        if self.transform:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = self.transform(img)

        return img, poses