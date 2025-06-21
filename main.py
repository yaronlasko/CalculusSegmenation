from os.path import join

import cv2
from matplotlib import pyplot as plt

from segment_teeth import YoloToothSegmenter
from pathlib import Path
import os
import yaml
from unet_overlay import plot_unet_predictions

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



def run_segmentation():
    segmenter = YoloToothSegmenter("default.yaml")
    segmenter.run()


def main():
   #run_segmentation()
   #plot_one_tooth_test()
   plot_unet_predictions()

if __name__ == "__main__":
    main()
