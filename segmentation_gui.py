import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import numpy as np
import cv2
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import os
import yaml
import sys

IMG_SIZE = (256, 256)
PADDING = 20
CONFIG_PATH = "default.yaml"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class SegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculus Segmentation")
        self.root.configure(bg="#e6efff")
        self.output_img = None
        self.display_percent = tk.BooleanVar(value=True)

        # Load models
        with open(resource_path(CONFIG_PATH), 'r') as f:
            config = yaml.safe_load(f)
        self.yolo_model = YOLO(resource_path(config['MODEL']['WEIGHTS']))
        self.unet_model = smp.Unet(encoder_name="resnet50", in_channels=3, classes=1).to(DEVICE)
        self.unet_model.load_state_dict(torch.load(resource_path(config['UNET']['WEIGHTS']), map_location=DEVICE))
        self.unet_model.eval()

        self.root.iconbitmap(resource_path("icon/tooth_icon.ico"))

        # Header
        tk.Label(root, text="🦷 Calculus Segmentation", font=("Helvetica", 20, "bold"), bg="#e6efff", fg="#003366").pack(pady=(20, 5))
        tk.Label(root, text="Yearly Project — Software Engineering", font=("Helvetica", 12), bg="#e6efff", fg="#003366").pack(pady=(0, 20))

        # Upload and Save Buttons Frame
        button_frame = tk.Frame(root, bg="#e6efff")
        button_frame.pack()

        self.upload_btn = tk.Button(button_frame, text="Upload Image", command=self.upload_image, bg="#007bff", fg="white", width=15, height=2)
        self.upload_btn.grid(row=0, column=0, padx=10)

        self.save_btn = tk.Button(button_frame, text="Save Result", command=self.save_image, bg="#007bff", fg="white", width=15, height=2)
        self.save_btn.grid(row=0, column=1, padx=10)
        self.save_btn.config(state=tk.DISABLED)

        self.percent_checkbox = tk.Checkbutton(root, text="Show Coverage Percentages", variable=self.display_percent, bg="#e6efff")
        self.percent_checkbox.pack(pady=(10, 5))

        self.image_panel = tk.Label(root, bg="#e6efff")
        self.image_panel.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if not file_path:
            return

        original = cv2.imread(file_path)
        result = self.run_segmentation(original)
        self.output_img = result

        image_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        img_pil.thumbnail((800, 800))
        img_tk = ImageTk.PhotoImage(img_pil)

        self.image_panel.configure(image=img_tk)
        self.image_panel.image = img_tk
        self.save_btn.config(state=tk.NORMAL)

    def run_segmentation(self, original_image):
        h, w, _ = original_image.shape
        results = self.yolo_model.predict(source=original_image, save=False, imgsz=640, conf=0.25, device=DEVICE.type)

        overlay = original_image.copy()
        annotations = np.zeros_like(overlay)

        for r in results:
            if r.masks is None:
                continue
            for mask in r.masks.data:
                mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                ys, xs = np.where(mask_np > 0)
                if ys.size == 0 or xs.size == 0:
                    continue
                y1, y2 = max(ys.min() - PADDING, 0), min(ys.max() + PADDING, h)
                x1, x2 = max(xs.min() - PADDING, 0), min(xs.max() + PADDING, w)

                tooth_crop = original_image[y1:y2, x1:x2]
                resized = cv2.resize(tooth_crop, IMG_SIZE) / 255.0
                input_tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    pred = self.unet_model(input_tensor)[0][0].cpu().numpy()

                pred_resized = cv2.resize(pred, (x2 - x1, y2 - y1))
                pred_mask = (pred_resized > 0.5).astype(np.uint8) * 255

                red_mask = np.zeros_like(tooth_crop)
                red_mask[:, :, 2] = pred_mask
                blended = cv2.addWeighted(tooth_crop, 1.0, red_mask, 0.5, 0)
                overlay[y1:y2, x1:x2] = blended

                if self.display_percent.get():
                    tooth_area = np.count_nonzero(mask_np[y1:y2, x1:x2])
                    overlap = np.count_nonzero((mask_np[y1:y2, x1:x2] > 0) & (pred_mask > 0))
                    percent = 100 * overlap / (tooth_area + 1e-6)

                    text = f"{percent:.1f}%"
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
                    tx = cx - tw // 2
                    ty = cy + th // 2

                    cv2.putText(annotations, text, (tx, ty), font, font_scale, (0, 0, 255), 1, cv2.LINE_8)

        text_mask = np.any(annotations != 0, axis=-1)
        overlay[text_mask] = annotations[text_mask]
        return overlay

    def save_image(self):
        if self.output_img is None:
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            cv2.imwrite(file_path, self.output_img)
            messagebox.showinfo("Saved", f"Result saved to {file_path}")

if __name__ == '__main__':
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()
