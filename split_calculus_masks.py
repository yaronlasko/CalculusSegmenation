import os
import json
import numpy as np
from pycocotools import mask as mask_utils
from os.path import join
import cv2

def create_calculus_masks(coco_path, image_name, crop_boxes, output_dir, indices=None):
    with open(coco_path) as f:
        coco_data = json.load(f)

    img_entry = next((img for img in coco_data['images'] if image_name in img['file_name']), None)
    if img_entry is None:
        return

    img_id = img_entry['id']
    anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

    os.makedirs(output_dir, exist_ok=True)

    for idx, (xmin, ymin, xmax, ymax) in enumerate(crop_boxes):
        tooth_idx = indices[idx] if indices else idx + 1
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        tooth_mask = np.zeros((height, width), dtype=np.uint8)

        for ann in anns:
            for seg in ann['segmentation']:
                shifted_seg = []
                for i in range(0, len(seg), 2):
                    x, y = seg[i], seg[i + 1]
                    x_rel = x - xmin
                    y_rel = y - ymin
                    if 0 <= x_rel < width and 0 <= y_rel < height:
                        shifted_seg.extend([x_rel, y_rel])
                if len(shifted_seg) < 6:
                    continue

                rle = mask_utils.frPyObjects([shifted_seg], height, width)
                bin_mask = mask_utils.decode(rle)[:, :, 0]

                # 👇 Fix added here
                if bin_mask.shape != (height, width):
                    bin_mask = cv2.resize(bin_mask, (width, height), interpolation=cv2.INTER_NEAREST)

                tooth_mask = np.logical_or(tooth_mask, bin_mask)

        tooth_mask = (tooth_mask * 255).astype(np.uint8)
        mask_path = join(output_dir, f"calculus_mask_{tooth_idx}.png")
        cv2.imwrite(mask_path, tooth_mask)