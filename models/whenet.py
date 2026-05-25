import keras
import numpy as np
import cv2
from utils_when import softmax, draw_axis as utils_draw_axis

# Try several possible EfficientNet import paths (different packages/versions)
# try:
    # efficientnet (package that provides tfkeras wrapper)
from efficientnet.tfkeras import EfficientNetB0
# print("Using EfficientNetB0 from efficientnet.tfkeras")
# except Exception:
#     try:
#         # older packaging
#         from efficientnet.keras import EfficientNetB0
#         print("Using EfficientNetB0 from efficientnet.keras")
#     except Exception:
#         try:
#             # TensorFlow Keras builtin (TF >= 2.5)
#             from tensorflow.keras.applications import EfficientNetB0
#             print("Using EfficientNetB0 from tensorflow.keras.applications")
#         except Exception:
#             try:
#                 # keras.applications (standalone Keras)
#                 from keras.applications.efficientnet import EfficientNetB0
#                 print("Using EfficientNetB0 from keras.applications.efficientnet")
#             except Exception:
#                 raise ImportError(
#                     "Could not import EfficientNetB0. Install one of: 'efficientnet', 'efficientnet.tfkeras', or use TensorFlow 2.x so that 'tensorflow.keras.applications.EfficientNetB0' is available."
#                 )


class WHENet:
    def __init__(self, snapshot=None):
        # avoid attempting to download imagenet weights from the internet here
        base_model = EfficientNetB0(include_top=False, weights=None, input_shape=(224, 224, 3))
        out = base_model.output
        out = keras.layers.GlobalAveragePooling2D()(out)
        fc_yaw = keras.layers.Dense(name='yaw_new', units=120)(out) # 3 * 120 = 360 degrees in yaw
        fc_pitch = keras.layers.Dense(name='pitch_new', units=66)(out)
        fc_roll = keras.layers.Dense(name='roll_new', units=66)(out)
        self.model = keras.models.Model(inputs=base_model.input, outputs=[fc_yaw, fc_pitch, fc_roll])
        if snapshot!=None:
            self.model.load_weights(snapshot)
        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = np.array(self.idx_tensor, dtype=np.float32)
        self.idx_tensor_yaw = [idx for idx in range(120)]
        self.idx_tensor_yaw = np.array(self.idx_tensor_yaw, dtype=np.float32)

    def get_angle(self, img):
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = np.array(img).astype(np.float32)

        # Ensure batch dimension and resize to model's expected input (224x224)
        if img.ndim == 4:
            b, h, w, c = img.shape
            if (h, w) != (224, 224):
                resized = np.zeros((b, 224, 224, c), dtype=np.float32)
                for i in range(b):
                    resized[i] = cv2.resize(img[i], (224, 224), interpolation=cv2.INTER_AREA)
                img = resized
        elif img.ndim == 3:
            h, w, c = img.shape
            if (h, w) != (224, 224):
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(img, 0)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        img = img / 255.0
        img = (img - mean) / std
        predictions = self.model.predict(img, batch_size=8)
        yaw_predicted = softmax(predictions[0])
        pitch_predicted = softmax(predictions[1])
        roll_predicted = softmax(predictions[2])
        yaw_predicted = np.sum(yaw_predicted*self.idx_tensor_yaw, axis=1)*3-180
        pitch_predicted = np.sum(pitch_predicted * self.idx_tensor, axis=1) * 3 - 99
        roll_predicted = np.sum(roll_predicted * self.idx_tensor, axis=1) * 3 - 99
        return yaw_predicted, pitch_predicted, roll_predicted

    def predict(self, img):
        """Compatibility wrapper: accepts HxWxC image or batch and returns scalar angles."""
        arr = img
        if isinstance(arr, np.ndarray) and arr.ndim == 3:
            arr = np.expand_dims(arr, 0)
        yaw, pitch, roll = self.get_angle(arr)
        # return first item as scalars for single-image input
        return float(yaw[0]), float(pitch[0]), float(roll[0])

    def draw_axis(self, img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
        """Compatibility wrapper to draw axes on an image using utils_when.draw_axis."""
        return utils_draw_axis(img, yaw, pitch, roll, tdx=tdx, tdy=tdy, size=size)