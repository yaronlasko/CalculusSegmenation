import torch
import cv2
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
from ultralytics import YOLO
import segmentation_models_pytorch as smp
from PIL import Image
import yaml

# ==== CONFIGURATION ====
CONFIG_PATH = "default.yaml"
IMAGE_PATH = "dataset/train/41B_JPG.rf.10c8e5f226b69020f216cb54a22364e5.jpg"  # <<-- Set this
IMG_SIZE = (256, 256)
PADDING = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_unet_predictions():
    # ==== LOAD CONFIG ====
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    yolo_model = YOLO(config['MODEL']['WEIGHTS'])
    UNET_MODEL_WEIGHTS = config['UNET']['WEIGHTS']

    # ==== LOAD UNET ====
    unet_model = smp.Unet(encoder_name="resnet50", in_channels=3, classes=1).to(DEVICE)
    unet_model.load_state_dict(torch.load(UNET_MODEL_WEIGHTS, map_location=DEVICE))
    unet_model.eval()

    # ==== SEGMENT TEETH USING YOLO ====
    original_image = cv2.imread(IMAGE_PATH)
    h, w, _ = original_image.shape
    results = yolo_model.predict(source=IMAGE_PATH, save=False, imgsz=640, conf=0.25, device=DEVICE.type)

    overlay = original_image.copy()
    colors = (0, 0, 255)

    for i, r in enumerate(results):
        if r.masks is None:
            print(f"No masks found in {IMAGE_PATH}")
            continue

        for j, mask in enumerate(r.masks.data):
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            ys, xs = np.where(mask_np > 0)
            if ys.size == 0 or xs.size == 0:
                continue

            y1, y2 = max(ys.min() - PADDING, 0), min(ys.max() + PADDING, h)
            x1, x2 = max(xs.min() - PADDING, 0), min(xs.max() + PADDING, w)

            tooth_crop = original_image[y1:y2, x1:x2]
            resized_crop = cv2.resize(tooth_crop, IMG_SIZE) / 255.0
            input_tensor = torch.tensor(resized_crop, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred = unet_model(input_tensor)[0][0].cpu().numpy()

            pred_resized = cv2.resize(pred, (x2 - x1, y2 - y1))
            pred_mask = (pred_resized > 0.5).astype(np.uint8) * 255

            # Overlay red prediction mask
            red_mask = np.zeros_like(tooth_crop)
            red_mask[:, :, 2] = pred_mask  # red channel

            blended = cv2.addWeighted(tooth_crop, 1.0, red_mask, 0.5, 0)
            overlay[y1:y2, x1:x2] = blended

    # ==== SHOW OUTPUT ====
    output_path = Path("predicted_from_train.png")
    cv2.imwrite(str(output_path), overlay)
    print(f"Prediction result saved to: {output_path}")


