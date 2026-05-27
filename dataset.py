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
    
fa_device = 'cuda' if torch.cuda.is_available() else 'cpu'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=fa_device)

# class I2HeadDataset(Dataset):
#     def __init__(self, dataset_dir, transform=None):
#         self.transform = transform
        
#         self.image_paths = []
#         self.mat_paths = []

#         for filename in os.listdir(dataset_dir):
#             if filename.endswith(".png"):
#                 img_path = os.path.join(dataset_dir, filename)
#                 mat_path = img_path.replace(".png", ".mat")
                
#                 if os.path.exists(mat_path):
#                     self.image_paths.append(img_path)
#                     self.mat_paths.append(mat_path)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         mat_path = self.mat_paths[idx]

#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         pose_mat = sio.loadmat(mat_path)
        
#         pose_vector = pose_mat["HP_camera"][0][3:] 
#         pose = torch.tensor(pose_vector, dtype=torch.float32)

#         if self.transform:
#             if not isinstance(img, Image.Image):
#                 img = Image.fromarray(img)
#             img = self.transform(img)

#         return img, pose

class I2HeadDataset(Dataset):
    def __init__(self, image_paths_or_dir, transform=None):
        """
        image_paths_or_dir: either a list of image paths or a path to a directory
        containing images (jpg/png). If a directory is provided, all image files
        inside it will be used (sorted alphabetically).
        """
        self.transform = transform

        if isinstance(image_paths_or_dir, str) and os.path.isdir(image_paths_or_dir):
            dataset_dir = image_paths_or_dir
            self.image_paths = [
                os.path.join(dataset_dir, f)
                for f in sorted(os.listdir(dataset_dir))
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
        else:
            # Expect an iterable of paths
            self.image_paths = list(image_paths_or_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # 1. Load Image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Find and load corresponding .mat file
        base_path, _ = os.path.splitext(img_path)
        mat_path = base_path + ".mat"
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Missing .mat file for image: {mat_path}")

        mat_data = scipy.io.loadmat(mat_path)
        vector_6d = mat_data['HP_camera'][0]

        # 3. Extract angles (originally in degrees)
        angle_1 = vector_6d[3]
        angle_2 = vector_6d[4]
        angle_3 = vector_6d[5]

        # 4. Convert degrees to radians to match your pipeline
        rad_1 = angle_1 * (np.pi / 180.0)
        rad_2 = angle_2 * (np.pi / 180.0)
        rad_3 = angle_3 * (np.pi / 180.0)

        # 5. ALIGN ORDER: Must be [Pitch, Yaw, Roll]
        pitch, yaw, roll = rad_1, rad_2, rad_3

        # Create a float32 tensor (casting from mat's float64 prevents type errors)
        poses = torch.tensor([pitch, yaw, roll], dtype=torch.float32)

        # Ensure transform receives a PIL Image (consistent with LP300W)
        if self.transform:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = self.transform(img)

        return img, poses