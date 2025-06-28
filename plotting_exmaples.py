from pathlib import Path
import os
import yaml
from unet_overlay import plot_unet_predictions
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt
from pycocotools import mask as mask_utils
from os.path import join

import cv2
from matplotlib import pyplot as plt

config_path = "default.yaml"


def plot_one_tooth_test():
    """
    this function plots a single tooth and it's mask from HARDCODED path
    """
    # Change to the image and tooth you want to test
    image_name = "IMG_4185_JPG.rf.2880017715d622a49a90719873b3aaa1"
    tooth_index = 1  # 1-based index
    relative_folder = "valid"

    config = None
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(config['DATA']['OUTPUT_DIR'])

    # Paths
    base_dir = output_dir / relative_folder / image_name
    tooth_path = join(base_dir,"data", f"tooth_{tooth_index}.png")
    mask_path = join(base_dir, "calculus_masks", f"calculus_mask_{tooth_index:02}.png")
    overlay_path = join(base_dir, f"overlay_{tooth_index}.png")

    data_dir = output_dir / relative_folder / image_name / 'data'
    teeth_mask_dir = output_dir / relative_folder / image_name / 'teeth_masks'

    # Load images
    tooth = cv2.imread(tooth_path)
    mask = cv2.imread(mask_path, 0)  # grayscale mask

    #guarantees match in size
    if mask.shape != tooth.shape[:2]:
        mask = cv2.resize(mask, (tooth.shape[1], tooth.shape[0]), interpolation=cv2.INTER_NEAREST)

    print("tooth.shape:", tooth.shape)
    print("mask.shape:", mask.shape)

    # Create red overlay where mask is positive
    overlay = tooth.copy()
    overlay[mask > 0] = [0, 0, 255]

    # Blend for visualization
    blended = cv2.addWeighted(tooth, 0.7, overlay, 0.3, 0)

    # Save overlay image
    cv2.imwrite(overlay_path, blended)
    print(f"Saved overlay to: {overlay_path}")

    # Show with matplotlib
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Tooth Image")
    plt.imshow(cv2.cvtColor(tooth, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Overlayed Calculus Mask")
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def plot_whole_image_with_mask():
    coco_path = "dataset/train/_annotations.coco.json"
    image_path = "dataset/train/41B_JPG.rf.10c8e5f226b69020f216cb54a22364e5.jpg"

    # Load COCO annotations
    with open(coco_path) as f:
        coco = json.load(f)

    image_filename = image_path.split("/")[-1]
    img_entry = next((img for img in coco['images'] if image_filename in img['file_name']), None)
    if img_entry is None:
        print(f"Image {image_filename} not found in COCO file.")
        return

    img_id = img_entry['id']
    width, height = img_entry['width'], img_entry['height']
    anns = [ann for ann in coco['annotations'] if ann['image_id'] == img_id]

    full_mask = np.zeros((height, width), dtype=np.uint8)

    for ann in anns:
        for seg in ann['segmentation']:
            if len(seg) < 6:
                continue
            rle = mask_utils.frPyObjects([seg], height, width)
            bin_mask = mask_utils.decode(rle)[:, :, 0]
            full_mask = np.logical_or(full_mask, bin_mask)

    full_mask = (full_mask * 255).astype(np.uint8)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    image = cv2.resize(image, (width, height))
    overlay = image.copy()
    overlay[full_mask > 0] = [0, 255, 0]  # Green overlay
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    # Save the output image
    cv2.imwrite("real.png", blended)