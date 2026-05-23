import os
import scipy.io as sio
import cv2
import numpy as np
import random
import torch
from tqdm import tqdm
from models.utils import compute_euler_angles_from_rotation_matrices

def set_global_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

###################################
############ Training #############
###################################

def evaluate_mae(model, dataloader, device):
    model.eval()
    
    total_mae_yaw = 0.0
    total_mae_pitch = 0.0
    total_mae_roll = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for imgs, poses in tqdm(dataloader, desc="Evaluando"):
            imgs = imgs.to(device)
            poses = poses.to(device)
            predictions = model(imgs)

            pred_mat = predictions[0]
            pred_angles = compute_euler_angles_from_rotation_matrices(pred_mat)
            pred_angles = pred_angles.to(device)
            total_mae_pitch += torch.sum(torch.abs(pred_angles[:, 0] - poses[:, 0])).item()
            total_mae_yaw += torch.sum(torch.abs(pred_angles[:, 1] - poses[:, 1])).item()
            total_mae_roll += torch.sum(torch.abs(pred_angles[:, 2] - poses[:, 2])).item()

            num_samples += imgs.size(0)
            
    mae_yaw = total_mae_yaw / num_samples
    mae_pitch = total_mae_pitch / num_samples
    mae_roll = total_mae_roll / num_samples
    mae_total = (mae_yaw + mae_pitch + mae_roll) / 3.0
    
    return mae_yaw, mae_pitch, mae_roll, mae_total

###################################
############ Occlusion ############
###################################

def extract_landmarks(ruta_imagen):
    photo_name = os.path.splitext(os.path.basename(ruta_imagen))[0]
    folder = os.path.basename(os.path.dirname(ruta_imagen))
    ruta_mat = f"./dataset/300W_LP/landmarks/{folder}/{photo_name}_pts.mat"
    
    if not os.path.exists(ruta_mat):
        raise FileNotFoundError(f"Error: No landmarks found for: {ruta_mat}")

    mat_data = sio.loadmat(ruta_mat)

    if 'pts_2d' in mat_data:
        landmarks = mat_data['pts_2d']
    else:
        raise ValueError(f"Error: No landmarks found for: {ruta_mat}")

    return landmarks.astype(np.int32)

def apply_occlusion(img, landmarks, mode="random"):
    """
    mode can be: 'random', 'clean', 'mask', 'glasses' or 'both'
    """
    img_occluded = img.copy()
    
    if mode == "random":
        choice = np.random.choice(["clean", "mask", "glasses", "both"], p=[0.25, 0.25, 0.25, 0.25])
    else:
        choice = mode
        
    if choice == "clean":
        return img_occluded

    if choice in ["mask", "both"]:
        jaw_points = landmarks[2:15]
        nose_points = landmarks[29:31] 

        mask_points = np.vstack((jaw_points, nose_points))

        hull = cv2.convexHull(mask_points)

        cv2.fillPoly(img_occluded, [hull], (0, 0, 0))

    if choice in ["glasses", "both"]:
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        left_hull = cv2.convexHull(left_eye)
        right_hull = cv2.convexHull(right_eye)
        
        cv2.fillPoly(img_occluded, [left_hull], (0, 0, 0))
        cv2.fillPoly(img_occluded, [right_hull], (0, 0, 0))
        
        expansion_thickness = 25
        cv2.polylines(img_occluded, [left_hull], isClosed=True, color=(0, 0, 0), thickness=expansion_thickness)
        cv2.polylines(img_occluded, [right_hull], isClosed=True, color=(0, 0, 0), thickness=expansion_thickness)
        
        pt1 = tuple(landmarks[39])
        pt2 = tuple(landmarks[42])
        cv2.line(img_occluded, pt1, pt2, (0, 0, 0), thickness=8)

    # for debugging
    # for idx, point in enumerate(landmarks):
    #     cv2.circle(img_occluded, tuple(point), 2, (0, 255, 0), -1)
    #     #mostrar texto con el indice del punto
    #     cv2.putText(img_occluded, str(idx), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    return img_occluded