import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch
import scipy.io as sio
import utils
import cv2    

class LP300W(Dataset):
    def __init__(self, image_paths, transform=None, apply_occlusion=True, occlusion_mode="random"):
        """
        image_paths: List of file paths to images
        occlusion_mode: 'random', 'clean', 'mask', 'glasses', or 'both'
        apply_occlusion: If False, ignores occlusion pipeline completely
        """
        self.occlusion_mode = occlusion_mode
        self.transform = transform
        self.apply_occlusion = apply_occlusion
        
        self.image_paths = []
        self.mat_paths = []

        for img_path in image_paths:
            mat_path = img_path.replace(".jpg", ".mat")
            if os.path.exists(mat_path):
                self.image_paths.append(img_path)
                self.mat_paths.append(mat_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mat_path = self.mat_paths[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pose_mat = sio.loadmat(mat_path)
        pose = torch.tensor(pose_mat["Pose_Para"][0][:3], dtype=torch.float32) # Yaw, Pitch, Roll

        if self.apply_occlusion:
            landmarks = utils.extract_landmarks(img_path)
            img = utils.apply_occlusion(img, landmarks, mode=self.occlusion_mode)

        if self.transform:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = self.transform(img)

        return img, pose