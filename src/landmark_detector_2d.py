import os
import numpy as np
import face_alignment
import matplotlib.pyplot as plt

class LandmarkDetector2D:
    """
    Thin wrapper over face-alignment for 68-point 2D detection.
    Expects an RGB uint8 image of shape (H,W,3).
    """

    def __init__(self, image_rgb_uint8, device="cpu"):
        self.image = image_rgb_uint8
        dev = "cuda" if (device.startswith("cuda") or device == "cuda") else "cpu"
        self.detector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, device=dev
        )

    def detect(self):
        landmarks = self.detector.get_landmarks_from_image(self.image)
        if not landmarks:
            return None
        return landmarks[0]  # (68,2) float

    def visualize(self, landmarks_2d):
        plt.imshow(self.image)
        if landmarks_2d is not None:
            plt.scatter(landmarks_2d[:, 0], landmarks_2d[:, 1], s=10)
        plt.title("Detected 2D Landmarks")
        plt.axis("off")
        plt.show()

    def save_landmarks(self, landmarks_2d, save_dir, save_name):
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, save_name if save_name.endswith(".npy") else f"{save_name}.npy")
        np.save(out, landmarks_2d)
        print(f"[2D] saved {out}")
