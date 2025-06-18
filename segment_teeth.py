from ultralytics import YOLO
import yaml
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from split_calculus_masks import create_calculus_masks
from os.path import join
import time



class YoloToothSegmenter:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_path = self.config['MODEL']['WEIGHTS']
        self.input_dir = Path(self.config['DATA']['INPUT_DIR'])
        self.output_dir = Path(self.config['DATA']['OUTPUT_DIR'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = YOLO(self.model_path)

    # def seg_calulus_all(self,crop_boxes):
    #     input_dir = Path(self.config['DATA']['INPUT_DIR'])
    #     coco_path = self.input_dir / "_annotations.coco.json"
    #
    #     # create_calculus_masks(coco_path, image_name, crop_boxes, output_dir,saved_teeth)
    #
    #
    #     image_paths = sorted(self.input_dir.glob("*.jpg"))
    #
    #     for image_path in image_paths:
    #         image_name = image_path.stem #name without .jpg
    #         output_masks_dir = join(self.output_dir, image_name, "calculus_masks")
    #
    #         create_calculus_masks(coco_path, image_name, crop_boxes, output_masks_dir)

    def run(self):
        image_paths = sorted(self.input_dir.glob("*.jpg"))
        PADDING = 20
        coco_path = self.input_dir / "_annotations.coco.json"

        for image_path in image_paths:
            image_name = image_path.stem
            print(f'[{time.strftime("%y%m%d-%H:%M:%S", time.localtime(time.time()))}] Processing {image_name}...')

            # Output structure
            relative_folder = image_path.parent.name
            data_dir = self.output_dir / relative_folder / image_name / 'data'
            teeth_mask_dir = self.output_dir / relative_folder / image_name / 'teeth_masks'
            data_dir.mkdir(parents=True, exist_ok=True)
            teeth_mask_dir.mkdir(parents=True, exist_ok=True)


            crop_boxes=[]

            # Run YOLOv8 segmentation
            device = "cuda" if torch.cuda.is_available() else "cpu"
            results = self.model.predict(source=str(image_path), save=False, imgsz=640, conf=0.25, device=device)
            image = cv2.imread(str(image_path))
            h, w, _ = image.shape

            for i, r in enumerate(results):
                if r.masks is None:
                    print(f"No masks found in {image_name}")
                    continue

                for j, mask in enumerate(r.masks.data):
                    mask_np = mask.cpu().numpy().astype(np.uint8) * 255

                    # Find bounding box from mask
                    ys, xs = np.where(mask_np > 0)
                    if ys.size == 0 or xs.size == 0:
                        continue
                    y1, y2 = max(ys.min() - PADDING, 0), min(ys.max() + PADDING, h)
                    x1, x2 = max(xs.min() - PADDING, 0), min(xs.max() + PADDING, w)

                    crop_boxes.append((x1, y1, x2, y2))  # saving box in list for future use

                    # Crop image and save
                    tooth_crop = image[y1:y2, x1:x2]
                    mask_crop = mask_np[y1:y2, x1:x2]

                    cv2.imwrite(str(data_dir / f"tooth_{j+1}.png"), tooth_crop)
                    cv2.imwrite(str(teeth_mask_dir / f"tooth_{j+1}_mask.png"), mask_crop)

            output_masks_dir = join(self.output_dir, relative_folder,image_name, "calculus_masks")
            create_calculus_masks(coco_path, image_name, crop_boxes, output_masks_dir)
            #break #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

