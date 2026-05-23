import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch
import scipy.io as sio
import utils
import cv2

class LP300W(Dataset):
    def __init__(self, root_dir, transform=None, apply_occlusion=True, occlusion_mode="clean"):
        """
        occlusion_mode: 'random', 'clean', 'mask', or 'glasses'
        apply_occlusion: If False, ignores occlusion pipeline
        """
        self.root_dir = root_dir
        self.mode = occlusion_mode
        self.transform = transform
        self.apply_occlusion = apply_occlusion
        self.image_paths = []
        self.mat_paths = []

        # Gather file pairs across subfolders
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for file in os.listdir(folder_path):
                if file.endswith(".jpg"):
                    img_path = os.path.join(folder_path, file)
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
            img = utils.apply_geometric_occlusion(img, landmarks, mode=self.mode)

        if self.transform:
            img = self.transform(img)

        return img, pose