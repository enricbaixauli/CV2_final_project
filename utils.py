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
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

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

def evaluate_metrics(model, dataloader, device):
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, poses in tqdm(dataloader, desc="Evaluating"):
            imgs = imgs.to(device)
            poses = poses.to(device)
            predictions = model(imgs)
            pred_mat = predictions[0]

            pred_angles = compute_euler_angles_from_rotation_matrices(pred_mat)

            all_preds.append(pred_angles.cpu().numpy())
            all_targets.append(poses.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    mae_array = mean_absolute_error(all_targets, all_preds, multioutput='raw_values')
    rmse_array = root_mean_squared_error(all_targets, all_preds, multioutput='raw_values')
    r2_array = r2_score(all_targets, all_preds, multioutput='raw_values')
    
    axes = ['pitch', 'yaw', 'roll']
    
    mae = {axes[i]: mae_array[i] for i in range(3)}
    mae['total'] = np.mean(mae_array)
    
    rmse = {axes[i]: rmse_array[i] for i in range(3)}
    rmse['total'] = np.mean(rmse_array)
    
    r2 = {axes[i]: r2_array[i] for i in range(3)}
    r2['total'] = np.mean(r2_array)
    
    return mae, rmse, r2

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

def plot_training_history(train_losses, val_losses, model_name="Model"):
    """
    Plots the training and validation loss curves for a single model run.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(9, 5))
    
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', linewidth=2, color='#1f77b4')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s', linewidth=2, color='#ff7f0e')
    
    plt.title(f'{model_name} - Training & Validation Loss History', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MAE in Radians)', fontsize=12)
    
    plt.xticks(epochs)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.show()    


def evaluate_paired_images_thpe(img_path1, img_path2, model_thpe, device=None):
    """
    Runs TokenHPE inference on two related images (e.g., raw vs. occluded),
    using the .mat file from the first image as the shared ground truth.
    Prints a comparative evaluation table and displays both images side-by-side.
    """
    # Safety checks for files
    if not os.path.exists(img_path1):
        print(f"Error: Base image path '{img_path1}' does not exist.")
        return
    if not os.path.exists(img_path2):
        print(f"Error: Modified image path '{img_path2}' does not exist.")
        return

    # 1. Ground Truth Extraction (Anchored to img_path1)
    base_path1, _ = os.path.splitext(img_path1)
    mat_path = base_path1 + ".mat"
    
    if not os.path.exists(mat_path):
        print(f"Error: Shared ground-truth file '{mat_path}' not found.")
        return
        
    mat = sio.loadmat(mat_path)
    vector_6d = mat["HP_camera"][0]
    
    gt_pitch = float(vector_6d[3])
    gt_yaw   = float(vector_6d[4])
    gt_roll  = float(vector_6d[5])

    # Load image matrices
    img_cv1 = cv2.imread(img_path1)
    img_cv2 = cv2.imread(img_path2)
    img_pil1 = Image.open(img_path1).convert("RGB")
    img_pil2 = Image.open(img_path2).convert("RGB")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    model_thpe.eval()
    
    # 2. Inference Pipeline (Image 1)
    x1 = transform(img_pil1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_mat1, _ = model_thpe(x1)
    euler1 = compute_euler_angles_from_rotation_matrices(pred_mat1)
    if isinstance(euler1, torch.Tensor): euler1 = euler1.cpu().numpy()
    euler1 = euler1 * 180.0 / np.pi
    p1, y1, r1 = float(euler1[0, 0]), float(euler1[0, 1]), float(euler1[0, 2])

    # 3. Inference Pipeline (Image 2)
    x2 = transform(img_pil2).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_mat2, _ = model_thpe(x2)
    euler2 = compute_euler_angles_from_rotation_matrices(pred_mat2)
    if isinstance(euler2, torch.Tensor): euler2 = euler2.cpu().numpy()
    euler2 = euler2 * 180.0 / np.pi
    p2, y2, r2 = float(euler2[0, 0]), float(euler2[0, 1]), float(euler2[0, 2])

    # 4. Error Calculations
    def get_error(pred, gt):
        if 'compute_angle_error' in globals():
            return compute_angle_error(pred, gt)
        return abs(pred - gt)
        
    # Image 1 Metrics
    err_p1 = get_error(p1, gt_pitch)
    err_y1 = get_error(y1, gt_yaw)
    err_r1 = get_error(r1, gt_roll)
    mae1 = (err_p1 + err_y1 + err_r1) / 3.0
    
    # Image 2 Metrics
    err_p2 = get_error(p2, gt_pitch)
    err_y2 = get_error(y2, gt_yaw)
    err_r2 = get_error(r2, gt_roll)
    mae2 = (err_p2 + err_y2 + err_r2) / 3.0
    
    # 5. Print Comparative Table
    print("\n" + "="*85)
    print(f"Ground Truth   | Pitch: {gt_pitch:+6.2f}° | Yaw: {gt_yaw:+6.2f}° | Roll: {gt_roll:+6.2f}°")
    print("="*85)
    print(f"{'Image Version':<16} | {'Pitch (Error)':<15} | {'Yaw (Error)':<15} | {'Roll (Error)':<15} | {'MAE':<7}")
    print("="*85)
    print(f"{'1. Base Image':<16} | {p1:+6.2f} ({err_p1:5.2f}°) | {y1:+6.2f} ({err_y1:5.2f}°) | {r1:+6.2f} ({err_r1:5.2f}°) | {mae1:5.2f}°")
    print(f"{'2. Mod Image':<16} | {p2:+6.2f} ({err_p2:5.2f}°) | {y2:+6.2f} ({err_y2:5.2f}°) | {r2:+6.2f} ({err_r2:5.2f}°) | {mae2:5.2f}°")
    print("="*85 + "\n")
    
    # 6. Axis Visualizations
    img_out1 = img_cv1.copy()
    draw_axis(img_out1, y1, p1, r1, size=100)
    
    img_out2 = img_cv2.copy()
    draw_axis(img_out2, y2, p2, r2, size=100)
    
    # 7. Side-by-Side Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    
    # Plot Image 1
    axes[0].imshow(cv2.cvtColor(img_out1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(
        f"1. Base Image (TokenHPE Predictions)\n"
        f"MAE: {mae1:.2f}°\n"
        f"Err -> P: {err_p1:.1f}°, Y: {err_y1:.1f}°, R: {err_r1:.1f}°",
        fontsize=11, fontweight='bold'
    )
    axes[0].axis('off')
    
    # Plot Image 2
    axes[1].imshow(cv2.cvtColor(img_out2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"2. Modified Image (TokenHPE Predictions)\n"
        f"MAE: {mae2:.2f}°\n"
        f"Err -> P: {err_p2:.1f}°, Y: {err_y2:.1f}°, R: {err_r2:.1f}°",
        fontsize=11, fontweight='bold'
    )
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()