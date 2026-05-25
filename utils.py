import os
import scipy.io as sio
import cv2
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from models.utils_hpe import compute_euler_angles_from_rotation_matrices, compute_rotation_matrix_from_ortho6d, draw_axis

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
        for imgs, poses in tqdm(dataloader, desc="Evaluating"):
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

def compute_angle_error(pred, gt):
    """Computes the absolute angular difference, correctly handling the 360-degree wrap-around."""
    diff = pred - gt
    return abs((diff + 180) % 360 - 180)

def compare_head_pose_models_with_error(img_path, model_thpe, model_sixD, model_whenet):
    """Runs inference across all pipelines, extracts ground truth labels, and outputs error metrics."""
    if not os.path.exists(img_path):
        print(f"Error: Target image file path '{img_path}' does not exist.")
        return

    # Ground Truth Extraction
    mat_path = img_path.replace(".jpg", ".mat")
    if not os.path.exists(mat_path):
        print(f"Error: Accompanying ground-truth file '{mat_path}' not found.")
        return
        
    mat = sio.loadmat(mat_path)
    gt_pose = mat["Pose_Para"][0][:3] * 180 / np.pi
    gt_pitch, gt_yaw, gt_roll = float(gt_pose[0]), float(gt_pose[1]), float(gt_pose[2])

    img_cv = cv2.imread(img_path)
    img_pil = Image.open(img_path).convert("RGB")
    
    predictions = {}
    
    # TokenHPE Inference 
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    x = transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        pred_rotation_matrix, _ = model_thpe(x)
        
    euler = compute_euler_angles_from_rotation_matrices(pred_rotation_matrix) * 180 / np.pi
    thpe_pitch = float(euler[0, 0])
    thpe_yaw   = float(euler[0, 1])
    thpe_roll  = float(euler[0, 2])
    predictions['TokenHPE'] = (thpe_pitch, thpe_yaw, thpe_roll)
    
    #  SixDRepNet Inference 
    sixd_pitch, sixd_yaw, sixd_roll = model_sixD.predict(img_cv.copy())
    sixd_pitch, sixd_yaw, sixd_roll = float(sixd_pitch), float(sixd_yaw), float(sixd_roll)
    predictions['SixDRepNet'] = (sixd_pitch, sixd_yaw, sixd_roll)
    
    # WHENet Inference 
    whenet_yaw, whenet_pitch, whenet_roll = model_whenet.predict(img_cv.copy())
    whenet_yaw, whenet_pitch, whenet_roll = float(whenet_yaw), float(whenet_pitch), float(whenet_roll)

    predictions['WHENet'] = (whenet_pitch, whenet_yaw, whenet_roll)
    
   
    print("\n" + "="*85)
    print(f"Ground Truth | Pitch: {gt_pitch:+6.2f}° | Yaw: {gt_yaw:+6.2f}° | Roll: {gt_roll:+6.2f}°")
    print("="*85)
    print(f"{'Model Identifier':<16} | {'Pitch (Error)':<15} | {'Yaw (Error)':<15} | {'Roll (Error)':<15} | {'MAE':<7}")
    print("="*85)
    
    errors = {}
    for model_name, (p, y, r) in predictions.items():
        err_p = compute_angle_error(p, gt_pitch)
        err_y = compute_angle_error(y, gt_yaw)
        err_r = compute_angle_error(r, gt_roll)
        mae = (err_p + err_y + err_r) / 3.0
        errors[model_name] = (err_p, err_y, err_r, mae)
        
        print(f"{model_name:<16} | {p:+6.2f} ({err_p:5.2f}°) | {y:+6.2f} ({err_y:5.2f}°) | {r:+6.2f} ({err_r:5.2f}°) | {mae:5.2f}°")
    print("="*85 + "\n")
    
    img_out_thpe = img_cv.copy()
    draw_axis(img_out_thpe, thpe_yaw, thpe_pitch, thpe_roll, size=100)
    
    img_out_sixD = img_cv.copy()
    model_sixD.draw_axis(img_out_sixD, sixd_yaw, sixd_pitch, sixd_roll)
    
    img_out_whenet = img_cv.copy()
    model_whenet.draw_axis(img_out_whenet, whenet_yaw, whenet_pitch, whenet_roll)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    
    images = [img_out_thpe, img_out_sixD, img_out_whenet]
    titles = [
        f"TokenHPE\nMAE: {errors['TokenHPE'][3]:.2f}°\nErr P:{errors['TokenHPE'][0]:.1f}°, Y:{errors['TokenHPE'][1]:.1f}°, R:{errors['TokenHPE'][2]:.1f}°",
        f"SixDRepNet\nMAE: {errors['SixDRepNet'][3]:.2f}°\nErr P:{errors['SixDRepNet'][0]:.1f}°, Y:{errors['SixDRepNet'][1]:.1f}°, R:{errors['SixDRepNet'][2]:.1f}°",
        f"WHENet\nMAE: {errors['WHENet'][3]:.2f}°\nErr P:{errors['WHENet'][0]:.1f}°, Y:{errors['WHENet'][1]:.1f}°, R:{errors['WHENet'][2]:.1f}°"
    ]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()