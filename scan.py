import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLO
import torch
import json
import requests
import shutil
import sys
import serial
import serial.tools.list_ports
import threading
import cv2
import time
import os
from datetime import datetime
import numpy as np

# BULK.PY Functions

LABEL_COLORS = {
    "Multiple_Narrow": "orange",
    "Transverse_Wide": "red",
    "Longitudinal_Wide": "blue",
    "Longitudinal_Narrow": "blue",
    "Transverse_Narrow": "red"
}

TYPE_SEVERITY = {
    "Multiple_Narrow": {"type": "Multiple", "severity": "Narrow"},
    "Transverse_Wide": {"type": "Transverse", "severity": "Wide"},
    "Longitudinal_Wide": {"type": "Longitudinal", "severity": "Wide"},
    "Longitudinal_Narrow": {"type": "Longitudinal", "severity": "Narrow"},
    "Transverse_Narrow": {"type": "Transverse", "severity": "Narrow"}
}

def parse_filename(filename):
    base_name = filename.split('.')[0]
    parts = base_name.split('-')
    
    if len(parts) != 5:
        raise ValueError("Filename doesn't match expected format")
    
    timestamp_str = parts[0].replace('_', ' ')
    date_part, time_part = timestamp_str.split()
    formatted_time = f"{time_part[:2]}-{time_part[2:4]}-{time_part[4:6]}"
    timestamp = f"{date_part}_{formatted_time}"
    
    try:
        lat1 = parts[1].replace('_', '.')
        long1 = parts[2].replace('_', '.')
        lat2 = parts[3].replace('_', '.')
        long2 = parts[4].replace('_', '.')
    except ValueError:
        raise ValueError("Invalid coordinate format")
    
    return {
        'timestamp': timestamp,
        'lat1': lat1,
        'long1': long1,
        'lat2': lat2,
        'long2': long2
    }

def load_model(model_path):
    return YOLO(model_path)

def resize_image(image, max_size=(900, 600)):  
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def save_image(img, original_path, directory):
    os.makedirs(directory, exist_ok=True)
    filename = os.path.basename(original_path)
    save_path = os.path.join(directory, filename)
    img.save(save_path)
    return save_path

def upload_imageonapi(file_path):
    try:
        with open(file_path, 'rb') as img_file:
            response = requests.post("https://api.arcdem.site/upload", files={'file': img_file})
        
        if response.status_code == 200:
            print(f"Image {os.path.basename(file_path)} uploaded successfully!")
            return True
        else:
            print(f"Upload failed for {os.path.basename(file_path)}. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error uploading {os.path.basename(file_path)}: {str(e)}")
        return False


# def process_image(image_path, model):
#     img = Image.open(image_path).convert("RGB")
#     img_width, img_height = img.size

#     results = model(img)  # Single result object
#     labels = list(LABEL_COLORS.keys())  

#     cracks_data = []
#     draw = ImageDraw.Draw(img)

#     try:
#         font = ImageFont.truetype("arial.ttf", 26)  
#     except IOError:
#         font = ImageFont.load_default()

#     result = results[0] if isinstance(results, list) else results  # support both single or list result

#     boxes = result.boxes.xywh.cpu().numpy() if torch.is_tensor(result.boxes.xywh) else result.boxes.xywh.numpy()
#     class_ids = result.boxes.cls.cpu().numpy().astype(int) if torch.is_tensor(result.boxes.cls) else result.boxes.cls.numpy().astype(int)
#     confidences = result.boxes.conf.cpu().numpy() if torch.is_tensor(result.boxes.conf) else result.boxes.conf.numpy()

#     for i, box in enumerate(boxes):
#         confidence = confidences[i]
#         if confidence < 0.0:
#             continue

#         label_id = class_ids[i]
#         label = result.names[label_id]  

#         if label in labels:
#             color = LABEL_COLORS.get(label, "red")  
#             x_center, y_center, width, height = box
#             x1, y1, x2, y2 = x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2

#             draw.rectangle([x1, y1, x2, y2], outline=color, width=4)  
#             text_position = (x1, max(y1 - 30, 5))  
#             text = f"{label} ({confidence:.2f})"
#             draw.text(text_position, text, fill=color, font=font)

#             width_conv = (width * 5) / img_height
#             height_conv = (height * 5) / img_height

#             index = len(cracks_data) + 1
#             extra_labels = []

#             if label in ["Transverse_Wide", "Transverse_Narrow"]:
#                 crack_info = {
#                     "type": get_type(label).lower(),
#                     "severity": get_severity(label).lower(),
#                     "length": str(round(width_conv, 2)),
#                     "index": index
#                 }
#                 extra_labels.append(f"L: {round(width_conv, 2)}m")

#             elif label in ["Longitudinal_Wide", "Longitudinal_Narrow"]:
#                 crack_info = {
#                     "type": get_type(label).lower(),
#                     "severity": get_severity(label).lower(),
#                     "length": str(round(height_conv, 2)),
#                     "index": index
#                 }
#                 extra_labels.append(f"L: {round(height_conv, 2)}m")

#             else:  # Multiple_Narrow
#                 crack_info = {
#                     "type": get_type(label).lower(),
#                     "severity": get_severity(label).lower(),
#                     "length": str(round(height_conv, 2)),
#                     "width": str(round(width_conv, 2)),
#                     "index": index
#                 }
#                 extra_labels.append(f"L: {round(height_conv, 2)}m")
#                 extra_labels.append(f"W: {round(width_conv, 2)}m")

#             extra_labels.append(f"Index: {index}")

#             for j, ltxt in enumerate(extra_labels):
#                 draw.text((x1, text_position[1] + 25 * (j + 1)), ltxt, fill="black", font=font)

#             cracks_data.append(crack_info)

#     filename = os.path.basename(image_path)
#     try:
#         parsed = parse_filename(filename)
#         result_data = {
#             "filename": filename.replace(".jpg", ""),
#             "start_coor": [parsed['lat1'], parsed['long1']],
#             "end_coor": [parsed['lat2'], parsed['long2']],
#             "date_created": parsed['timestamp'],
#             "cracks": cracks_data
#         }
#     except ValueError:
#         result_data = {
#             "filename": filename,
#             "start_coor": None,
#             "end_coor": None,
#             "date_created": datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S"),
#             "cracks": cracks_data
#         }

#     return cracks_data, result_data, img

# def process_image(image_path, model):
#     img = Image.open(image_path).convert("RGB")
#     img_width, img_height = img.size

#     results = model(img, conf=0.30, iou=0.25)  # Assuming model processes image and returns results
#     labels = list(LABEL_COLORS.keys())  

#     cracks_data = []
#     draw = ImageDraw.Draw(img)

#     try:
#         font = ImageFont.truetype("arial.ttf", 24)  # Larger font
#     except IOError:
#         font = ImageFont.load_default()

#     # Padding values (adjust as needed)
#     padding_top = 10
#     padding_left = 10

#     for result in results:
#         boxes = result.boxes.xywh.cpu().numpy() if torch.is_tensor(result.boxes.xywh) else result.boxes.xywh.numpy()
#         class_ids = result.boxes.cls.cpu().numpy().astype(int) if torch.is_tensor(result.boxes.cls) else result.boxes.cls.numpy().astype(int)
#         confidences = result.boxes.conf.cpu().numpy() if torch.is_tensor(result.boxes.conf) else result.boxes.conf.numpy()

#         for i, box in enumerate(boxes):
#             confidence = confidences[i]
#             if confidence < 0.0:
#                 continue

#             label_id = class_ids[i]
#             label = result.names[label_id]

#             if label in labels:
#                 color = LABEL_COLORS.get(label, "red")  
#                 x_center, y_center, width, height = box
#                 x1, y1, x2, y2 = x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2

#                 draw.rectangle([x1, y1, x2, y2], outline=color, width=4)  # Draw bounding box
                
#                 # Calculate index and text lines
#                 index = len(cracks_data) + 1
#                 conf_val = round(confidence, 2)
#                 width_val = round(width * 5 / img_height, 2)  # Assuming 5 meters for conversion
#                 height_val = round(height * 5 / img_height, 2)
                
#                 label_text = "#" + str(index) + " " + str(label) + " (" + str(conf_val) + ")"
#                 length_text = "L: " + str(round(width_val if 'Transverse' in label else height_val, 2)) + "m"
#                 width_text = "W: " + str(round(width_val, 2)) + "m" if "Multiple_Narrow" in label else ""

#                 text_lines = [label_text, length_text]
#                 if width_text:
#                     text_lines.append(width_text)

#                 # Calculate spacing above the bounding box with padding
#                 line_spacing = 32  # Adjust as needed
#                 total_text_height = len(text_lines) * line_spacing
#                 start_y = max(y1 - total_text_height - padding_top, 5)  # Ensure space above the box

#                 # Draw text above the box with padding
#                 for j, line in enumerate(text_lines):
#                     # Set black color for length and width text
#                     fill_color = "black" if j > 0 else color
#                     draw.text((x1 + padding_left, start_y + j * line_spacing), line, fill=fill_color, font=font)

#                 # Add crack data to list
#                 crack_info = {
#                     "type": get_type(label).lower(),
#                     "severity": get_severity(label).lower(),
#                     "length": str(round(width_val if 'Transverse' in label else height_val, 2)),
#                     "width": str(round(width_val, 2)),
#                     "index": index
#                 }
#                 cracks_data.append(crack_info)

#     filename = os.path.basename(image_path)
#     try:
#         parsed = parse_filename(filename)
#         result_data = {
#             "filename": filename.replace(".jpg", ""),
#             "start_coor": [parsed['lat1'], parsed['long1']],
#             "end_coor": [parsed['lat2'], parsed['long2']],
#             "date_created": parsed['timestamp'],
#             "cracks": cracks_data
#         }
#     except ValueError:
#         result_data = {
#             "filename": filename,
#             "start_coor": None,
#             "end_coor": None,
#             "date_created": datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S"),
#             "cracks": cracks_data
#         }

#     return cracks_data, result_data, img

def process_image(image_path, model):
    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size

    results = model(img, conf=0.1, iou=0.3)
    labels = list(LABEL_COLORS.keys())  

    cracks_data = []
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    padding_top = 10
    padding_left = 10

    for result in results:
        boxes = result.boxes.xywh.cpu().numpy() if torch.is_tensor(result.boxes.xywh) else result.boxes.xywh.numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int) if torch.is_tensor(result.boxes.cls) else result.boxes.cls.numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy() if torch.is_tensor(result.boxes.conf) else result.boxes.conf.numpy()

        box_data = []
        for i, box in enumerate(boxes):
            confidence = confidences[i]
            if confidence < 0.0:
                continue

            x_center, y_center, width, height = box
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            area = width * height

            box_data.append({
                "index": i,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": width,
                "height": height,
                "x_center": x_center,
                "y_center": y_center,
                "confidence": confidence,
                "class_id": class_ids[i],
                "label": result.names[class_ids[i]],
                "area": area
            })

        # SMART FILTER: Remove boxes ≥90% inside any bigger box (regardless of class)
        keep_indices = set(range(len(box_data)))
        for i in range(len(box_data)):
            for j in range(len(box_data)):
                if i == j:
                    continue
                box_i = box_data[i]
                box_j = box_data[j]

                ix1 = max(box_i["x1"], box_j["x1"])
                iy1 = max(box_i["y1"], box_j["y1"])
                ix2 = min(box_i["x2"], box_j["x2"])
                iy2 = min(box_i["y2"], box_j["y2"])

                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                intersection_area = iw * ih

                if intersection_area / box_i["area"] >= 0.5 and box_j["area"] > box_i["area"]:
                    keep_indices.discard(box_i["index"])

        # Now only process boxes that passed the filter
        for i, box_info in enumerate(box_data):
            if i not in keep_indices:
                continue

            label = box_info["label"]
            if label in labels:
                color = LABEL_COLORS.get(label, "red")  
                x1, y1, x2, y2 = box_info["x1"], box_info["y1"], box_info["x2"], box_info["y2"]

                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

                index = len(cracks_data) + 1
                conf_val = round(box_info["confidence"], 2)
                width_val = round(box_info["width"] * 5 / img_height, 2)
                height_val = round(box_info["height"] * 5 / img_height, 2)

                label_text = "#" + str(index) + " " + str(label) + " (" + str(conf_val) + ")"
                length_text = "L: " + str(round(width_val if 'Transverse' in label else height_val, 2)) + "m"
                width_text = "W: " + str(round(width_val, 2)) + "m" if "Multiple_Narrow" in label else ""

                text_lines = [label_text, length_text]
                if width_text:
                    text_lines.append(width_text)

                line_spacing = 32
                total_text_height = len(text_lines) * line_spacing
                start_y = max(y1 - total_text_height - padding_top, 5)

                for j, line in enumerate(text_lines):
                    fill_color = "black" if j > 0 else color
                    draw.text((x1 + padding_left, start_y + j * line_spacing), line, fill=fill_color, font=font)

                crack_info = {
                    "type": get_type(label).lower(),
                    "severity": get_severity(label).lower(),
                    "length": str(round(width_val if 'Transverse' in label else height_val, 2)),
                    "width": str(round(width_val, 2)),
                    "index": index
                }
                cracks_data.append(crack_info)

    filename = os.path.basename(image_path)
    try:
        parsed = parse_filename(filename)
        result_data = {
            "filename": filename.replace(".jpg", ""),
            "start_coor": [parsed['lat1'], parsed['long1']],
            "end_coor": [parsed['lat2'], parsed['long2']],
            "date_created": parsed['timestamp'],
            "cracks": cracks_data
        }
    except ValueError:
        result_data = {
            "filename": filename,
            "start_coor": None,
            "end_coor": None,
            "date_created": datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S"),
            "cracks": cracks_data
        }

    return cracks_data, result_data, img

def upload_logs(data):
    url = "https://api.arcdem.site/update_logs"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=json.dumps([data]))
        if response.status_code == 200:
            print("Logs successfully uploaded!")
            return True
        else:
            print(f"Failed to upload logs. Status Code: {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        print(f"An error occurred during upload: {str(e)}")
        return False

def get_type(label):
    if label in TYPE_SEVERITY:
        return TYPE_SEVERITY[label]["type"]
    return "Unknown"

def get_severity(label):
    if label in TYPE_SEVERITY:
        return TYPE_SEVERITY[label]["severity"]
    return "Unknown"

def detect_images():
    process_videos_in_folder()  # <-- Process videos into panoramas first
    time.sleep(1)  # Delay after processing each video
    
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    panorama_dir = os.path.join(script_dir, "panoramas")
    panorama_done_dir = os.path.join(panorama_dir, "done")
    detect_image_dir = os.path.join(script_dir, "detections/img")
    detect_json_dir = os.path.join(script_dir, "detections/json")

    os.makedirs(panorama_done_dir, exist_ok=True)
    os.makedirs(detect_image_dir, exist_ok=True)
    os.makedirs(detect_json_dir, exist_ok=True)

    model = load_model("best.pt")
    image_files = [f for f in os.listdir(panorama_dir) if f.lower().endswith('.jpg')]
    
    if not image_files:
        messagebox.showinfo("Info", "No JPG images found in panoramas directory.")
        return

    for image_file in image_files:
        image_path = os.path.join(panorama_dir, image_file)
        try:
            cracks_data, result_data, img = process_image(image_path, model)
            save_image(img, image_path, detect_image_dir)

            json_filename = os.path.splitext(image_file)[0] + ".json"
            with open(os.path.join(detect_json_dir, json_filename), 'w') as f:
                json.dump(result_data, f, indent=2)

            shutil.move(image_path, os.path.join(panorama_done_dir, image_file))
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

    messagebox.showinfo("Info", f"Detection completed for {len(image_files)} images!")

def send_images():
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    detect_image_dir = os.path.join(script_dir, "detections/img")
    detect_json_dir = os.path.join(script_dir, "detections/json")
    sent_image_dir = os.path.join(script_dir, "sent/img")
    sent_json_dir = os.path.join(script_dir, "sent/json")

    os.makedirs(sent_image_dir, exist_ok=True)
    os.makedirs(sent_json_dir, exist_ok=True)

    image_files = [f for f in os.listdir(detect_image_dir) if f.lower().endswith('.jpg')]

    for image_file in image_files:
        try:
            image_path = os.path.join(detect_image_dir, image_file)
            json_path = os.path.join(detect_json_dir, os.path.splitext(image_file)[0] + ".json")

            if upload_imageonapi(image_path) and upload_logs(json.load(open(json_path))):
                shutil.move(image_path, os.path.join(sent_image_dir, image_file))
                shutil.move(json_path, os.path.join(sent_json_dir, os.path.basename(json_path)))
        except Exception as e:
            print(f"Error sending {image_file}: {str(e)}")

    messagebox.showinfo("Info", "Send process completed!")
    
def extract_frames(video_path, frame_interval=10, overlap=0.05):
    cap = cv2.VideoCapture(video_path)
    frames = []
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cropped_height = height - 0  # Crop bottom 22 pixels
    step = int(width * (1 - overlap))  # Step with slight overlap
    
    frame_count = 0
    success, frame = cap.read()
    last_frame = None
    
    while success:
        if frame_count % frame_interval == 0 or frame_count == 0:
            cropped_frame = frame[:cropped_height, :]
            frames.append(cropped_frame)
            print(f"Extracted frame {frame_count} from {os.path.basename(video_path)}")
        last_frame = frame
        frame_count += 1
        success, frame = cap.read()
    
    if last_frame is not None and (len(frames) == 0 or not np.array_equal(frames[-1], last_frame[:cropped_height, :])):
        frames.append(last_frame[:cropped_height, :])  # Ensure last frame is included
        print("Added last frame")
    
    cap.release()
    print(f"Total frames extracted: {len(frames)}")
    return frames

def stitch_frames(frames):
    print("Stitching process started...")
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, panorama = stitcher.stitch(frames)
    
    if status == cv2.Stitcher_OK:
        print("Stitching successful.")
        return panorama
    else:
        print(f"Error stitching frames. Status code: {status}")
        return None

def process_videos_in_folder():
    # folder_path = os.getcwd()  # Change to your folder path
    video_extensions = {'.avi', '.mp4', '.mov', '.mkv'}  # Supported formats
    input_folder = "./recordings"
    output_folder = "./panoramas"
    os.makedirs(output_folder, exist_ok=True)
    
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        
        if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in video_extensions:
            print(f"Processing video: {file_name}")
            frames = extract_frames(file_path)
            
            if len(frames) < 2:
                print("Not enough frames to stitch for", file_name)
                continue
            panorama = stitch_frames(frames)
            
            if panorama is not None:
                output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".jpg")
                cv2.imwrite(output_path, panorama)
                print(f"Panorama saved as {output_path}")
            else:
                print(f"Failed to create panorama for {file_name}")

# Globals
ser = None
ser2 = None
cap = None
recording = False
video_writer = None
frame_lock = threading.Lock()
current_frame = None
camera_running = True

lat = 0.0
lng = 0.0
gps_lat = None
gps_lng = None
gps_lat2 = None
gps_lng2 = None
gps_received_stage = 0  # 0 = waiting for lat, 1 = waiting for lng, 2 = waiting for second lat, 3 = waiting for second lng

# Directory to save videos
save_folder = "recordings"
os.makedirs(save_folder, exist_ok=True)

def view_panorama():
    image_folder = "./detections/img"
    json_folder = "./detections/json"
    trash_img_folder = "./detections/trash/img"
    trash_json_folder = "./detections/trash/json"

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)
    os.makedirs(trash_img_folder, exist_ok=True)
    os.makedirs(trash_json_folder, exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]

    if not image_files:
        messagebox.showinfo("No Images", "No images found in the directory.")
        return

    # === Create Dark Overlay ===
    overlay = tk.Toplevel(root)
    overlay.attributes('-fullscreen', True)
    overlay.attributes('-alpha', 0.5)
    overlay.configure(bg='black')
    overlay.transient(root)

    # === Modal Viewer Window ===
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    modal_height = int(screen_height * 0.9)
    modal_width = int(modal_height * 0.6)

    modal_x = (screen_width - modal_width) // 2
    modal_y = (screen_height - modal_height) // 2

    viewer = tk.Toplevel(overlay)
    viewer.geometry(f"{modal_width}x{modal_height}+{modal_x}+{modal_y}")
    viewer.title("Image Viewer")
    viewer.configure(bg="white")
    viewer.resizable(False, False)
    viewer.transient(overlay)
    viewer.grab_set()

    current_index = [0]

    image_display_width = int(modal_width * 0.75)
    image_display_height = int(modal_height * 0.9)

    img_label = ttk.Label(viewer, anchor="center", background="white")
    img_label.place(
        relx=0.5,
        rely=0.45,
        anchor="center",
        width=image_display_width,
        height=image_display_height
    )

    button_size = (modal_width - image_display_width) // 2 - 10
    button_y = int((modal_height * 0.45) - button_size // 2)
    button_font = ("Arial", 24)

    left_btn = tk.Button(
        viewer, text="◀", font=button_font,
        command=lambda: show_prev()
    )
    left_btn.place(x=10, y=button_y, width=button_size, height=button_size)

    right_btn = tk.Button(
        viewer, text="▶", font=button_font,
        command=lambda: show_next()
    )
    right_btn.place(x=modal_width - button_size - 10, y=button_y, width=button_size, height=button_size)

    delete_margin = 20
    delete_button_width = modal_width - (delete_margin * 2)

    delete_btn = ttk.Button(viewer, text="Delete", command=lambda: delete_current_image())
    delete_btn.place(x=delete_margin, y=int(modal_height * 0.93), width=delete_button_width, height=30)

    def show_image(index):
        img_path = os.path.join(image_folder, image_files[index])
        img = Image.open(img_path)
        img.thumbnail((image_display_width, image_display_height))
        photo = ImageTk.PhotoImage(img)

        img_label.configure(image=photo)
        img_label.image = photo
        viewer.title(f"{image_files[index]} ({index + 1}/{len(image_files)})")

    def show_next():
        if current_index[0] < len(image_files) - 1:
            current_index[0] += 1
            show_image(current_index[0])

    def show_prev():
        if current_index[0] > 0:
            current_index[0] -= 1
            show_image(current_index[0])

    def close_modal(event=None):
        if viewer.winfo_exists():
            viewer.destroy()
        if overlay.winfo_exists():
            overlay.destroy()

    def delete_current_image():
        index = current_index[0]
        image_filename = image_files[index]
        image_path = os.path.join(image_folder, image_filename)

        base_name = os.path.splitext(image_filename)[0]
        json_path = os.path.join(json_folder, base_name + ".json")

        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete:\n\nImage: {image_filename}\nJSON: {base_name}.json?\n\nFiles will be moved to trash.",
            parent=viewer
        )

        if confirm is not True:
            return

        try:
            if os.path.exists(image_path):
                shutil.move(image_path, os.path.join(trash_img_folder, image_filename))
            if os.path.exists(json_path):
                shutil.move(json_path, os.path.join(trash_json_folder, base_name + ".json"))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to move files to trash:\n{str(e)}", parent=viewer)
            return

        del image_files[index]

        if not image_files:
            messagebox.showinfo("No Images Left", "All images have been deleted.", parent=viewer)
            close_modal()
        else:
            if index >= len(image_files):
                current_index[0] = len(image_files) - 1
            show_image(current_index[0])

    # Escape + Overlay click = Close
    viewer.bind("<Escape>", close_modal)
    viewer.protocol("WM_DELETE_WINDOW", close_modal)
    overlay.bind("<Button-1>", lambda e: close_modal())

    show_image(current_index[0])
   
    
def update_entry(value):
    current_value = distance_var.get()
    distance_var.set(current_value + str(value))

def clear_entry():
    distance_var.set("")

def get_com_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

# def connect_serial():
#     global ser, ser2
#     port = com_var.get()
#     if not port:
#         messagebox.showerror("Error", "Please select a COM port.")
#         return
#     try:
#         ser = serial.Serial(port, 9600, timeout=1)
#         time.sleep(2)
#         status_var.set(f"Connected to {port}")
#         threading.Thread(target=listen_serial, daemon=True).start()
#     except Exception as e:
#         messagebox.showerror("Serial Error", str(e))

def connect_serial():
    global ser, ser2
    ports = list(serial.tools.list_ports.comports())

    for port in ports:
        try:
            s = serial.Serial(port.device, 9600, timeout=1)
            time.sleep(1)  # wait for Arduino UNO to reboot

            s.reset_input_buffer()  # clear any junk data
            s.write(b'\n')  # trigger response

            line = s.readline().decode(errors='ignore').strip()
            print(f"{port.device} says: {line}")

            if "[ID]MOTOR" in line and not ser:
                ser = s  # assign to motor
                print("MOTOR Arduino connected.")
            elif "[ID]GPS" in line and not ser2:
                ser2 = s  # assign to gps
                print("GPS Arduino connected.")
            else:
                s.close()
        except Exception as e:
            print(f"⚠️ Error on {port.device}: {e}")

    if not ser:
        print("❌ MOTOR Arduino not found.")
        return
    if not ser2:
        print("❌ GPS Arduino not found.")
        return
    
    print("Both Arduinos are successfuly connected!")
    threading.Thread(target=listen_serial, daemon=True).start()
    threading.Thread(target=update_gps, daemon=True).start()

def disconnect_serial():
    global ser, ser2
    if ser is not None:
        ser.close()  # Close the first serial connection
        print("Serial connection with motor Arduino closed.")
    if ser2 is not None:
        ser2.close()  # Close the second serial connection
        print("Serial connection with GPS Arduino closed.")
    
    ser, ser2 = None, None  # Set the serial objects to None


def update_gps():
    global lat, lng
    buffer = ""

    print("Update GPS thread started")
    while True:
        try:
            if ser2.in_waiting > 0:
                byte = ser2.read().decode(errors='ignore')
                if byte == '\n':
                    line = buffer.strip()
                    if line.startswith("LAT:"):
                        try:
                            lat = float(line[4:])
                        except ValueError:
                            pass
                    elif line.startswith("LNG:"):
                        try:
                            lng = float(line[4:])
                        except ValueError:
                            pass
                    print("Updated Coordinates:", lat, lng)
                    buffer = ""
                else:
                    buffer += byte
        except Exception as e:
            print("❌ GPS thread error:", e)
        time.sleep(0.01)


# def update_gps():
#     global lat, lng
#     buffer = ""

#     while True:
#         try:
#             if ser2 is not None and ser2.in_waiting > 0:
#                 byte = ser2.read().decode(errors='ignore')
#                 if byte == '\n':
#                     line = buffer.strip()
#                     if line.startswith("LAT:"):
#                         try:
#                             lat = float(line[4:])
#                         except ValueError:
#                             pass
#                     elif line.startswith("LNG:"):
#                         try:
#                             lng = float(line[4:])
#                         except ValueError:
#                             pass
#                     buffer = ""
#                 else:
#                     buffer += byte
#         except Exception as e:
#             print("❌ GPS thread error:", e)
#         time.sleep(0.01)


def send_segment():
    if ser and ser.is_open:
        segment = segment_var.get()
        if segment:
            ser.write(f"{segment}\n".encode())
    else:
        messagebox.showwarning("Warning", "Serial port not connected.")

# def listen_serial():
#     global recording, gps_lat, gps_lng, gps_lat2, gps_lng2, gps_received_stage
#     while True:
#         if ser and ser.in_waiting:
#             try:
#                 data = ser.readline().decode().strip()
#                 print("Received:", data)

#                 # Expecting two float values separately (first is latitude, then longitude)
#                 try:
#                     value = float(data)
#                     if gps_received_stage == 0:
#                         gps_lat = value
#                         gps_received_stage = 1
#                         print(f"[GPS] Latitude received: {gps_lat}")
#                     elif gps_received_stage == 1:
#                         gps_lng = value
#                         gps_received_stage = 2
#                         print(f"[GPS] Longitude received: {gps_lng}")
#                     elif gps_received_stage == 2:
#                         gps_lat2 = value
#                         gps_received_stage = 3
#                         print(f"[GPS] Second Latitude received: {gps_lat2}")
#                     elif gps_received_stage == 3:
#                         gps_lng2 = value
#                         gps_received_stage = 0  # Reset for next cycle
#                         print(f"[GPS] Second Longitude received: {gps_lng2}")
#                         # Now both lat/lng values are received, start recording
#                         start_recording()  # This will create the video after receiving lat2 and long2
#                 except ValueError:  
#                     if data == "START RECORD":
#                         start_recording()
#                     elif data == "END RECORD":
#                         stop_recording()
#             except Exception as e:
#                 print("Serial Read Error:", e)
#         time.sleep(0.1)

def listen_serial():
    global recording, gps_lat, gps_lng, gps_lat2, gps_lng2, gps_received_stage
    while True:
        if ser and ser.in_waiting:
            try:
                data = ser.readline().decode().strip()  
                print("MOTOR command:", data)

                if data == "START RECORD":
                    gps_lat = lat
                    gps_lng = lng
                    print(f"Starting record with coordinates: ({gps_lat}, {gps_lng})")
                    start_recording()

                elif data == "STOP RECORD":
                    gps_lat2 = lat
                    gps_lng2 = lng
                    print(f"Stopping record with coordinates: ({gps_lat2}, {gps_lng2})")
                    stop_recording()

            except Exception as e:
                print("Serial Read Error:", e)
        
        time.sleep(0.1)

def start_camera():
    global cap, camera_running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Failed to open camera.")
        return

    def show_feed():
        global current_frame
        while camera_running:
            ret, frame = cap.read()
            if not ret:
                continue
            with frame_lock:
                current_frame = frame.copy()
                    
                resized = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                height, width, _ = rgb_frame.shape
                cv2.line(rgb_frame, (0, 325), (width, 325), (255, 0, 0), 2)
                
                imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
                
                def update_gui():
                    camera_label.imgtk = imgtk
                    camera_label.configure(image=imgtk)

                camera_label.after(0, update_gui)
                time.sleep(0.03)  # Add slight delay to reduce flickering

    threading.Thread(target=show_feed, daemon=True).start()      

def start_recording():
    global video_writer, recording, gps_lat, gps_lng, gps_lat2, gps_lng2, video_filename
    if not cap or not cap.isOpened():
        return

    # Ensure gps_lat and gps_lng are from the first set of coordinates
    lat_str = str(gps_lat).replace(".", "_") if gps_lat is not None else "NA"
    lng_str = str(gps_lng).replace(".", "_") if gps_lng is not None else "NA"
    
    # Set initial filename with only the first set of coordinates (before receiving second set)
    gps_info = f"{lat_str}-{lng_str}-NA-NA"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(save_folder, f"{timestamp}-{gps_info}.avi")

    # Initialize the video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cropped_height = height - 78
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, cropped_height))

    if not video_writer.isOpened():
        print("❌ Failed to open VideoWriter. Check codec or path.")
        return

    # Start recording immediately after the first set of coordinates
    recording = True
    status_var.set("Recording")
    print(f"✅ Recording started: {video_filename}")

    def record_loop():
        global recording
        while recording:
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                    cropped = frame[:cropped_height, :]
                    video_writer.write(cropped)
            time.sleep(0.05)

    threading.Thread(target=record_loop, daemon=True).start()

def stop_recording():
    global recording, video_writer, gps_lat, gps_lng, gps_lat2, gps_lng2, video_filename
    recording = False
    time.sleep(0.1)

    # Now, update the filename with the second set of coordinates
    if gps_lat2 is not None and gps_lng2 is not None:
        lat2_str = str(gps_lat2).replace(".", "_") if gps_lat2 is not None else "NA"
        lng2_str = str(gps_lng2).replace(".", "_") if gps_lng2 is not None else "NA"
        
        # Ensure that gps_lat and gps_lng are used in the final filename
        lat_str = str(gps_lat).replace(".", "_") if gps_lat is not None else "NA"
        lng_str = str(gps_lng).replace(".", "_") if gps_lng is not None else "NA"
        
        # Update the filename with the second set of coordinates
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = os.path.join(save_folder, f"{timestamp}-{lat_str}-{lng_str}-{lat2_str}-{lng2_str}.avi")

        # Release the video writer before renaming the file
        if video_writer:
            video_writer.release()
            video_writer = None
            print(f"✅ Video writer released, ready to rename the file.")

        # Wait for the file to be fully released before renaming it
        time.sleep(0.5)  # Small delay to ensure the file is fully released

        # Rename the video file to reflect the second set of coordinates
        try:
            os.rename(video_filename, final_filename)
            print(f"✅ Video saved as: {final_filename}")
        except OSError as e:
            print(f"❌ Error renaming file: {e}")

    status_var.set("Not Recording")
    print("🛑 Recording stopped.")


# def start_recording():
#     global video_writer, recording, gps_lat, gps_lng, gps_lat2, gps_lng2
#     if not cap or not cap.isOpened():
#         return

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cropped_height = height - 63

#     # Use both sets of latitudes and longitudes for the filename
#     lat_str = str(gps_lat).replace(".", "_") if gps_lat is not None else "NA"
#     lng_str = str(gps_lng).replace(".", "_") if gps_lng is not None else "NA"
#     lat2_str = str(gps_lat2).replace(".", "_") if gps_lat2 is not None else "NA"
#     lng2_str = str(gps_lng2).replace(".", "_") if gps_lng2 is not None else "NA"
#     gps_info = f"{lat_str}-{lng_str}-{lat2_str}-{lng2_str}"

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = os.path.join(save_folder, f"{timestamp}-{gps_info}.avi")

#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, cropped_height))

#     if not video_writer.isOpened():
#         print("❌ Failed to open VideoWriter. Check codec or path.")
#         return

#     recording = True
#     status_var.set("Recording")
#     print(f"✅ Recording started: {filename}")

#     def record_loop():
#         global recording
#         while recording:
#             with frame_lock:
#                 if current_frame is not None:
#                     frame = current_frame.copy()
#                     cropped = frame[:cropped_height, :]
#                     video_writer.write(cropped)
#             time.sleep(0.05)

#     threading.Thread(target=record_loop, daemon=True).start()

# def stop_recording():
#     global recording, video_writer
#     recording = False
#     time.sleep(0.1)
#     if video_writer:
#         video_writer.release()
#         video_writer = None
#         print("✅ Video saved.")
#     status_var.set("Not Recording")
#     print("🛑 Recording stopped.")

def on_close():
    global camera_running, cap, ser, recording
    camera_running = False
    recording = False
    if cap:
        cap.release()
    if ser and ser.is_open:
        ser.close()
    cv2.destroyAllWindows()
    root.destroy()
def send_segment():
    if ser and ser.is_open:
        segment = distance_var.get()
        if segment:
            ser.write(f"{segment}\n".encode())
    else:
        messagebox.showwarning("Warning", "Serial port not connected.")
# GUI Setup
bg_color = "#D2B746"

root = tk.Tk()
root.title("Segment Sender + Serial Camera Recorder")
root.geometry("1270x700")
root.resizable(True, True)

# Main Frames
left_frame = tk.Frame(root, width=640, bg=bg_color)
right_frame = tk.Frame(root, width=640, bg=bg_color)
left_frame.pack(side="left", fill="both", expand=True)
right_frame.pack(side="right", fill="both", expand=True)


# Header Label
header_label = tk.Label(left_frame, text="ARCDEM", font=("Arial Black", 28), fg="brown", bg=bg_color)
header_label.pack(pady=5)

# COM Port Instructions
ttk.Label(left_frame, text="Connect / Disconnect MOTOR and GPS COM Ports", font=("Arial", 10), background=bg_color).pack(pady=5)

# Connect / Disconnect Buttons Side by Side
conn_frame = tk.Frame(left_frame, bg=bg_color)
conn_frame.pack(pady=5)

tk.Button(conn_frame, text="Connect", command=connect_serial, bg="green", fg="white", font=("Arial", 12, "bold")).pack(side="left", padx=10)
tk.Button(conn_frame, text="Disconnect", command=disconnect_serial, bg="red", fg="white", font=("Arial", 12, "bold")).pack(side="left", padx=10)

# Distance Entry
distance_label = tk.Label(left_frame, text="Enter Distance:", font=("Arial", 12), bg=bg_color)
distance_label.pack(pady=5)
distance_var = tk.StringVar()
distance_frame = tk.Frame(left_frame, bg=bg_color)
distance_frame.pack(pady=5)

distance_entry = ttk.Entry(distance_frame, textvariable=distance_var, font=("Arial", 14), justify='center', width=20)
distance_entry.pack(side="left", padx=5)

# Clear Button inside the Entry
clear_button = tk.Button(distance_frame, text="X", command=clear_entry, font=("Arial", 14, "bold"), fg="white", bg="red")
clear_button.place(relx=0.98, rely=0.5, anchor="e", relheight=0.8)  # Adjusted position to be inside the Entry

# Numpad
numpad_frame = tk.Frame(left_frame, bg="dark gray")
numpad_frame.pack(pady=5)
buttons = [
    ('1', 1, 0), ('2', 1, 1), ('3', 1, 2),
    ('4', 2, 0), ('5', 2, 1), ('6', 2, 2),
    ('7', 3, 0), ('8', 3, 1), ('9', 3, 2),
    ('0', 4, 1), ('VIEW', 4, 0), ('RUN', 4, 2)
]
for text, row, col in buttons:
    btn = tk.Button(numpad_frame, text=text, command=(lambda t=text: update_entry(t)) if text.isdigit() else (view_panorama if text == 'VIEW' else lambda: send_segment()), font=("Arial", 14), bg="gray" if text.isdigit() else ("blue" if text == 'VIEW' else "green"), fg="white", width=5, height=2)
    btn.grid(row=row, column=col, padx=5, pady=5)

# Status
status_var = tk.StringVar(value="Not Recording")
ttk.Label(left_frame, textvariable=status_var, foreground="red", background=bg_color, font=("Arial", 14, "bold")).pack(pady=15)

# Detect / Send Buttons Side by Side
detect_send_frame = tk.Frame(left_frame, bg=bg_color)
detect_send_frame.pack(pady=5)

tk.Button(detect_send_frame, text="Detect", command=detect_images, bg="blue", fg="white", font=("Arial", 14, "bold")).pack(side="left", padx=10)
tk.Button(detect_send_frame, text="Send", command=send_images, bg="green", fg="white", font=("Arial", 14, "bold")).pack(side="left", padx=10)


# Right Frame: Camera Feed
camera_label = tk.Label(right_frame, bg=bg_color)
camera_label.pack(fill="both", expand=True)

# Camera Status
status_frame = tk.Frame(right_frame, bg=bg_color)
status_frame.place(relx=0.09, rely=0.86, anchor='sw')

status_text_label = tk.Label(status_frame, text="Status:", font=("Arial", 14, "bold"), bg=bg_color, fg="white")
status_text_label.pack(side="left")

camera_status_var = tk.StringVar(value="READY")
camera_status_label = tk.Label(status_frame, textvariable=camera_status_var, font=("Arial", 14, "bold"), bg=bg_color, fg="green")
camera_status_label.pack(side="left", padx=5)

# Coordinates at the bottom-right
coordinates_var = tk.StringVar(value=f"Coordinates: {lat}, {lng}")

coord_frame = tk.Frame(right_frame, bg=bg_color)
coord_frame.pack(side="bottom", anchor="se", padx=10, pady=10, fill="x")

coordinates_label = ttk.Label(coord_frame, textvariable=coordinates_var, font=("Arial", 11), background=bg_color, foreground="white")
coordinates_label.pack(anchor="e")  # Aligns to bottom-right

# Start webcam feed
start_camera()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()

