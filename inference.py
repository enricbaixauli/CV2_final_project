import argparse
import cv2
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
import utils
import matplotlib
import numpy as np
import seaborn as sns
from PIL import Image
from model import TokenHPE
sns.set()

matplotlib.use('TkAgg')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Predict using TokenHPE model')

    parser.add_argument('--model_path',
                        dest='model_path', help='model weights path',
                        default='./weights/TokenHPEv1-ViTB-224_224-lyr3.tar', type=str)
    parser.add_argument('--show_viz',
                        dest='show_viz', help='Save images with pose cube.',
                        default=True, type=bool)
    parser.add_argument('--image_path',
                        dest='image_path', help='image_path',
                        default="", type=str)
    parser.add_argument('--save_path', dest='save_path',
                        default='./output/vis/res.png',
                        help='prediction image save path', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    model_path = args.model_path


    model = TokenHPE(num_ori_tokens=9,
                 depth=3, heads=8, embedding='sine', dim=128, inference_view=True
                 ).to("cuda")


    print('Loading data...')

    transformations = transforms.Compose([transforms.Resize(270),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    print("Loading model...")
    if model_path != "":
        saved_state_dict = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in saved_state_dict:
            model.load_state_dict(saved_state_dict['model_state_dict'])
            print("model weight loaded!")
        else:
            model.load_state_dict(saved_state_dict)
    else:
        print("model weight failed!")

    model.to("cuda")

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    total = 0
    yaw_error = pitch_error = roll_error = .0
    v1_err = v2_err = v3_err = .0

    with torch.no_grad():

        img_path = args.image_path
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = transformations(img)
        img = torch.unsqueeze(img, dim=0)
        img = torch.Tensor(img).to("cuda")

        R_pred, ori_9_d = model(img)

        euler = utils.compute_euler_angles_from_rotation_matrices(
            R_pred) * 180 / np.pi
        p_pred_deg = euler[:, 0].cpu()
        y_pred_deg = euler[:, 1].cpu()
        r_pred_deg = euler[:, 2].cpu()
        print(f"Prediction: pitch:{p_pred_deg[0]:.2f}, yaw:{y_pred_deg[0]:.2f}, roll:{r_pred_deg[0]:.2f}.")

        if args.show_viz:
            # to show the predicted image
            cv2_img = cv2.imread(img_path)
            utils.draw_axis(cv2_img, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0],  size=100) # tdx=150, tdy=150,
            # utils.plot_pose_cube(cv2_img, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], size=100)
            cv2.imshow("Prediction", cv2_img)
            cv2.waitKey(0)

        # save image
        save_path = args.save_path
        cv2.imwrite(save_path, cv2_img)
        print("Image saved to: ", save_path)





