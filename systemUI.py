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
import serial.serialutil
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


# Globals
ser = None
ser2 = None
cap = None
recording = False
video_writer = None
frame_lock = threading.Lock()
current_frame = None
camera_running = True
ready = False  # Ensure this variable is defined and set properly in your script logic
status_var = None
connErr = 0

lat = 0.0
lng = 0.0
gps_lat = None
gps_lng = None
gps_lat2 = None
gps_lng2 = None
gps_received_stage = 0  # 0 = waiting for lat, 1 = waiting for lng, 2 = waiting for second lat, 3 = waiting for second lng

# Directory to save videos



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

def process_image(image_path, model):
    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size

    results = model(img, conf=0.20, iou=0.30)  # Assuming model processes image and returns results
    labels = list(LABEL_COLORS.keys())  

    cracks_data = []
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 24)  # Larger font
    except IOError:
        font = ImageFont.load_default()

    # Padding values (adjust as needed)
    padding_top = 10
    padding_left = 10

    for result in results:
        boxes = result.boxes.xywh.cpu().numpy() if torch.is_tensor(result.boxes.xywh) else result.boxes.xywh.numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int) if torch.is_tensor(result.boxes.cls) else result.boxes.cls.numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy() if torch.is_tensor(result.boxes.conf) else result.boxes.conf.numpy()

        for i, box in enumerate(boxes):
            confidence = confidences[i]
            if confidence < 0.0:
                continue

            label_id = class_ids[i]
            label = result.names[label_id]

            if label in labels:
                color = LABEL_COLORS.get(label, "red")  
                x_center, y_center, width, height = box
                x1, y1, x2, y2 = x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2

                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)  # Draw bounding box
                
                # Calculate index and text lines
                index = len(cracks_data) + 1
                conf_val = round(confidence, 2)
                width_val = round(width * 5 / img_height, 2)  # Assuming 5 meters for conversion
                height_val = round(height * 5 / img_height, 2)
                
                label_text = "#" + str(index) + " " + str(label) + " (" + str(conf_val) + ")"
                length_text = "L: " + str(round(width_val if 'Transverse' in label else height_val, 2)) + "m"
                width_text = "W: " + str(round(width_val, 2)) + "m" if "Multiple_Narrow" in label else ""

                text_lines = [label_text, length_text]
                if width_text:
                    text_lines.append(width_text)

                # Calculate spacing above the bounding box with padding
                line_spacing = 32  # Adjust as needed
                total_text_height = len(text_lines) * line_spacing
                start_y = max(y1 - total_text_height - padding_top, 5)  # Ensure space above the box

                # Draw text above the box with padding
                for j, line in enumerate(text_lines):
                    # Set black color for length and width text
                    fill_color = "black" if j > 0 else color
                    draw.text((x1 + padding_left, start_y + j * line_spacing), line, fill=fill_color, font=font)

                # Add crack data to list
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
    panorama_dir = "./panoramas"
    panorama_done_dir = "./panoramas/done"
    detect_image_dir = "./detections/img"
    detect_json_dir = "./detections/json"

    os.makedirs(panorama_done_dir, exist_ok=True)
    os.makedirs(detect_image_dir, exist_ok=True)
    os.makedirs(detect_json_dir, exist_ok=True)

    model = load_model("best.pt")

    while True:
        for image_file in os.listdir(panorama_dir):
            image_path = os.path.join(panorama_dir, image_file)

            if os.path.isdir(image_path):
                continue

            if "-NA-NA" in os.path.splitext(image_file)[0]:
                continue

            try:
                cracks_data, result_data, img = process_image(image_path, model)
                save_image(img, image_path, detect_image_dir)

                json_filename = os.path.splitext(image_file)[0] + ".json"
                with open(os.path.join(detect_json_dir, json_filename), 'w') as f:
                    json.dump(result_data, f, indent=2)

                shutil.move(image_path, os.path.join(panorama_done_dir, image_file))
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

        # messagebox.showinfo("Info", f"Detection completed for {len(image_files)} images!")
   
# def detect_image(filename):
#     detection_running = True
#     process_recording(filename)  # <-- Process videos into panoramas first
#     time.sleep(1)  # Delay after processing each video

#     panorama_dir = "./panoramas"
#     panorama_done_dir = "./panoramas/done"
#     detect_image_dir = "./detections/img"
#     detect_json_dir = "./detections/json"

#     os.makedirs(panorama_dir, exist_ok=True)
#     os.makedirs(panorama_done_dir, exist_ok=True)
#     os.makedirs(detect_image_dir, exist_ok=True)
#     os.makedirs(detect_json_dir, exist_ok=True)

#     model = load_model("best.pt")

#     image_path = os.path.join(panorama_dir, filename)
#     if not os.path.isfile(image_path):
#         messagebox.showinfo("Info", f"File not found: {filename}")
#         return

#     try:
#         cracks_data, result_data, img = process_image(image_path, model)
#         save_image(img, image_path, detect_image_dir)

#         json_filename = os.path.splitext(filename)[0] + ".json"
#         with open(os.path.join(detect_json_dir, json_filename), 'w') as f:
#             json.dump(result_data, f, indent=2)

#         shutil.move(image_path, os.path.join(panorama_done_dir, filename))
#     except Exception as e:
#         print(f"Error processing {filename}: {str(e)}")

#     detection_running = False


def send_images():
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    detect_image_dir = os.path.join(script_dir, "detections/img")
    detect_json_dir = os.path.join(script_dir, "detections/json")

    sent_image_dir = os.path.join(script_dir, "sent/img")
    sent_json_dir = os.path.join(script_dir, "sent/json")

    notsent_image_dir = os.path.join(script_dir, "notsent/img")
    notsent_json_dir = os.path.join(script_dir, "notsent/json")

    # Create all necessary folders
    os.makedirs(sent_image_dir, exist_ok=True)
    os.makedirs(sent_json_dir, exist_ok=True)
    os.makedirs(notsent_image_dir, exist_ok=True)
    os.makedirs(notsent_json_dir, exist_ok=True)

    image_files = [f for f in os.listdir(detect_image_dir) if f.lower().endswith('.jpg')]

    for image_file in image_files:
        try:
            image_path = os.path.join(detect_image_dir, image_file)
            json_filename = os.path.splitext(image_file)[0] + ".json"
            json_path = os.path.join(detect_json_dir, json_filename)

            if not os.path.exists(json_path):
                print(f"⚠️ Skipping {image_file} — JSON file not found.")
                shutil.move(image_path, os.path.join(notsent_image_dir, image_file))
                continue

            with open(json_path) as f:
                json_data = json.load(f)

            # Upload JSON first
            if upload_logs(json_data):
                print(f"✅ JSON uploaded: {json_filename}")
                # Now upload the image
                if upload_imageonapi(image_path):
                    print(f"✅ Image uploaded: {image_file}")
                    # Move both to sent
                    shutil.move(image_path, os.path.join(sent_image_dir, image_file))
                    shutil.move(json_path, os.path.join(sent_json_dir, json_filename))
                else:
                    print(f"❌ Image upload failed: {image_file}")
                    shutil.move(image_path, os.path.join(notsent_image_dir, image_file))
                    shutil.move(json_path, os.path.join(notsent_json_dir, json_filename))
            else:
                print(f"❌ JSON upload failed: {json_filename}")
                shutil.move(image_path, os.path.join(notsent_image_dir, image_file))
                shutil.move(json_path, os.path.join(notsent_json_dir, json_filename))

        except Exception as e:
            print(f"🔥 Error sending {image_file}: {str(e)}")
            try:
                if os.path.exists(image_path):
                    shutil.move(image_path, os.path.join(notsent_image_dir, image_file))
                if os.path.exists(json_path):
                    shutil.move(json_path, os.path.join(notsent_json_dir, json_filename))
            except Exception as move_error:
                print(f"🚫 Error moving failed files to notsent: {move_error}")

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

def process_recordings():
    video_extensions = {'.avi', '.mp4', '.mov', '.mkv'}  # Supported formats
    input_folder = "./recordings"
    output_folder = "./panoramas"
    done_folder = "./recordings/done"

    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(done_folder, exist_ok=True)

    while True:
        for file_name in os.listdir(input_folder):
            time.sleep(1)
            file_path = os.path.join(input_folder, file_name)

            if "-NA-NA" in os.path.splitext(file_name)[0]:
                continue

            if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in video_extensions:
                print(f"Processing video: {file_name}")
                frames = extract_frames(file_path)

                if len(frames) < 2:
                    print("Not enough frames to stitch for", file_name)
                    continue

                panorama = stitch_frames(frames)

                if panorama is not None:
                    image_file_name = os.path.splitext(file_name)[0] + ".jpg"
                    output_path = os.path.join(output_folder, image_file_name)
                    cv2.imwrite(output_path, panorama)
                    print(f"Panorama saved as {output_path}\n")

                    # Move the processed video to the 'done' folder
                    shutil.move(file_path, os.path.join(done_folder, file_name))
                    # time.sleep(.2)  # Delay after processing each video
                    # detect_image(image_file_name)

                else:
                    print(f"Failed to create panorama for {file_name}")

# def process_recording(filename):
#     video_extensions = {'.avi', '.mp4', '.mov', '.mkv'}  # Supported formats
#     input_folder = "./recordings"
#     output_folder = "./panoramas"
#     done_folder = "./recordings/done"

#     os.makedirs(input_folder, exist_ok=True)
#     os.makedirs(output_folder, exist_ok=True)
#     os.makedirs(done_folder, exist_ok=True)

#     file_path = os.path.join(input_folder, filename)

#     if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in video_extensions:
#         print(f"Processing video: {filename}")
#         frames = extract_frames(file_path)

#         if len(frames) < 2:
#             print(f"Not enough frames to stitch for {filename}")
#             return

#         panorama = stitch_frames(frames)

#         if panorama is not None:
#             output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".jpg")
#             cv2.imwrite(output_path, panorama)
#             print(f"Panorama saved as {output_path}\n")

#             # Move the processed video to the 'done' folder
#             shutil.move(file_path, os.path.join(done_folder, filename))
#         else:
#             print(f"Failed to create panorama for {filename}")
#     else:
#         print(f"File {filename} not found or unsupported format.")

def view_panorama():
    image_folder = "./detections/img"
    json_folder = "./detections/json"
    trash_img_folder = "./detections/trash/img"
    trash_json_folder = "./detections/trash/json"

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)
    os.makedirs(trash_img_folder, exist_ok=True)
    os.makedirs(trash_json_folder, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))], reverse=True)

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
        current_index[0] = (current_index[0] + 1) % len(image_files)
        show_image(current_index[0])

    def show_prev():
        current_index[0] = (current_index[0] - 1) % len(image_files)
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

def initialize_serial():
    global ready
    while not connect_serial():
        time.sleep(.3)

    ready = True

def connect_serial():
    global ser, ser2, ready, connErr
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
        connErr = 1
        return False
    if not ser2:
        print("❌ GPS Arduino not found.")
        connErr = 2
        return False
    
    print("Both Arduinos are successfuly connected!")
    threading.Thread(target=listen_serial, daemon=True).start()
    threading.Thread(target=update_gps, daemon=True).start()
    return True

def disconnect_serial():
    global ser, ser2, ready
    if ready:
        ready = False
        check_ready_state()
        update_status("Restarting")

    if ser is not None:
        ser.close()  # Close the first serial connection
        print("Serial connection with MOTOR Arduino closed.")
    if ser2 is not None:
        ser2.close()  # Close the second serial connection
        print("Serial connection with GPS Arduino closed.")
    
    ser, ser2 = None, None  # Set the serial objects to None

def restart_serial():
    def sequential_restart():
        global ready
        disconnect_serial()
        while not connect_serial():
            time.sleep(.3)

        time.sleep(3)
        ready = True
        check_ready_state()
        update_status("Ready to Run")

    # Run the sequence in one background thread
    threading.Thread(target=sequential_restart, daemon=True).start()

def update_gps():
    global lat, lng, ready
    buffer = ""
    last_restart = 0

    print("Update GPS thread started")
    while ser2 is not None:
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
            connErr = 2
            if ready:
                ready = False
                check_ready_state()
                update_status("GPS Arduino Disconnected")

                # 🔁 Restart serial connection safely
                if time.time() - last_restart > 2:  # prevent rapid loops
                    restart_serial()
                    last_restart = time.time()

        time.sleep(0.01)

def listen_serial():
    global recording, gps_lat, gps_lng, gps_lat2, gps_lng2, gps_received_stage, ready
    last_restart = 0
    stop_event = threading.Event()

    while ser is not None:
        try:
            if ser and ser.is_open and ser.in_waiting:
                data = ser.readline().decode().strip()  
                print("MOTOR command:", data)

                if data == "A":
                    gps_lat = lat
                    gps_lng = lng
                    print(f"Starting record with coordinates: ({gps_lat}, {gps_lng})")
                    start_recording()


                elif data == "B":
                    gps_lat2 = lat
                    gps_lng2 = lng
                    print(f"Stopping record with coordinates: ({gps_lat2}, {gps_lng2})")
                    stop_recording()

                elif data == "C":
                    enable_widgets(root)

        except serial.SerialException as e:
            print("❌ MOTOR thread error:", e)
            connErr = 1
            if ready:
                ready = False
                check_ready_state()
                update_status("MOTOR Arduino Disconnected")

                if time.time() - last_restart > 2:
                    restart_serial()
                    last_restart = time.time()
    
        except Exception as e:
            print("❌ MOTOR thread error:", e)
            update_status(f"Serial Error: {e}")

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

# def start_recording():
#     save_folder = "recordings"
#     os.makedirs(save_folder, exist_ok=True)

#     global video_writer, recording, gps_lat, gps_lng, gps_lat2, gps_lng2, video_filename
#     if not cap or not cap.isOpened():
#         return

#     # Ensure gps_lat and gps_lng are from the first set of coordinates
#     lat_str = str(gps_lat).replace(".", "_") if gps_lat is not None else "NA"
#     lng_str = str(gps_lng).replace(".", "_") if gps_lng is not None else "NA"
    
#     # Set initial filename with only the first set of coordinates (before receiving second set)
#     gps_info = f"{lat_str}-{lng_str}-NA-NA"

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     video_filename = os.path.join(save_folder, f"{timestamp}-{gps_info}.avi")

#     # Initialize the video writer
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cropped_height = height - 78
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, cropped_height))

#     if not video_writer.isOpened():
#         print("❌ Failed to open VideoWriter. Check codec or path.")
#         return

#     # Start recording immediately after the first set of coordinates
#     recording = True
#     camera_status_var.set("Recording")
#     print(f"✅ Recording started: {video_filename}")

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

def stop_recording(delete=False):
    save_folder = "recordings"
    global recording, video_writer, gps_lat, gps_lng, gps_lat2, gps_lng2, video_filename
    curr_filename = video_filename
    recording = False
    camera_status_var.set("Not Recording")
    print("🛑 Recording stopped.")
    time.sleep(0.01)

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
        time.sleep(1)  # Small delay to ensure the file is fully released

        if delete:
            if os.path.exists(curr_filename):
                os.remove(curr_filename)
                print(f"🗑️ Deleted: {curr_filename}")
            else:
                print(f"⚠️ File not found for deletion: {curr_filename}")
            return

        # Rename the video file to reflect the second set of coordinates
        try:
            os.rename(curr_filename, final_filename)
            print(f"✅ Video saved as: {final_filename}")
        except OSError as e:
            print(f"❌ Error renaming file: {e}")


    # threading.Thread(target=detect_image(final_filename), daemon=True).start()

def start_recording():
    global recording
    recording = True

def record_loop():
    global recording, video_writer, gps_lat, gps_lng, gps_lat2, gps_lng2
    global video_filename, cropped_height

    save_folder = "recordings"
    os.makedirs(save_folder, exist_ok=True)

    while True:
        if recording and (video_writer is None or not video_writer.isOpened()):
            if not cap or not cap.isOpened():
                print("❌ Camera not available.")
                recording = False
                continue

            lat_str = str(gps_lat).replace(".", "_") if gps_lat is not None else "NA"
            lng_str = str(gps_lng).replace(".", "_") if gps_lng is not None else "NA"
            gps_info = f"{lat_str}-{lng_str}-NA-NA"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = os.path.join(save_folder, f"{timestamp}-{gps_info}.avi")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cropped_height = height - 78
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, cropped_height))

            if not video_writer.isOpened():
                print("❌ Failed to open VideoWriter.")
                recording = False
                continue

            camera_status_var.set("Recording")
            print(f"✅ Recording started: {video_filename}")

        if recording and video_writer and video_writer.isOpened():
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                    cropped = frame[:cropped_height, :]
                    video_writer.write(cropped)

        time.sleep(0.05)

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

def run_segment():
    if ser and ser.is_open:
        segment = distance_var.get()
        if segment:
            # disable_widgets(root)
            ser.write(f"{segment}\n".encode())
    else:
        messagebox.showwarning("Warning", "Serial port not connected.")

def stop_segment():
    if ser and ser.is_open:
        ser.write(b"X\n")
        stop_recording(delete=True)

# threading.Thread(target=initialize_serial, daemon=True).start()
threading.Thread(target=process_recordings, daemon=True).start()
threading.Thread(target=detect_images, daemon=True).start()
threading.Thread(target=initialize_serial, daemon=True).start()

loading_overlay = None

def disable_widgets(widget):
    for child in widget.winfo_children():
        try:
            child.configure(state="disabled")
        except:
            pass
        disable_widgets(child)

def enable_widgets(widget):
    for child in widget.winfo_children():
        try:
            child.configure(state="normal")
        except:
            pass
        enable_widgets(child)

def show_loading_overlay():
    global loading_overlay
    if loading_overlay is None:
        loading_overlay = tk.Frame(root, bg='gray', width=1270, height=700)
        loading_overlay.place(relx=0.5, rely=0.5, anchor='center')

        loading_label = tk.Label(loading_overlay, text="Loading... Please wait", font=("Arial", 20, "bold"), fg="white", bg="gray")
        loading_label.pack(expand=True)

        disable_widgets(root)

def hide_loading_overlay():
    global loading_overlay
    if loading_overlay:
        loading_overlay.destroy()
        loading_overlay = None
        enable_widgets(root)

def check_ready_state():
    global ready
    if ready:
        hide_loading_overlay()
    else:
        show_loading_overlay()
        root.after(500, check_ready_state)


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

tk.Button(conn_frame, text="RESTART", command=restart_serial, bg="green", fg="white", font=("Arial", 12, "bold")).pack(side="left", padx=10)
# tk.Button(conn_frame, text="Disconnect", command=disconnect_serial, bg="red", fg="white", font=("Arial", 12, "bold")).pack(side="left", padx=10)

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
    btn = tk.Button(numpad_frame, text=text, command=(lambda t=text: update_entry(t)) if text.isdigit() else (view_panorama if text == 'VIEW' else lambda: run_segment()), font=("Arial", 14), bg="gray" if text.isdigit() else ("blue" if text == 'VIEW' else "green"), fg="white", width=5, height=2)
    btn.grid(row=row, column=col, padx=5, pady=5)

# Status
status_var = tk.StringVar(value="Initializing")
ttk.Label(left_frame, textvariable=status_var, foreground="red", background=bg_color, font=("Arial", 14, "bold")).pack(pady=15)
def update_status(label):
    root.after(0, lambda: status_var.set(label))

# Detect / Send Buttons Side by Side
detect_send_frame = tk.Frame(left_frame, bg=bg_color)
detect_send_frame.pack(pady=5)

tk.Button(detect_send_frame, text="Stop", command=stop_segment, bg="blue", fg="white", font=("Arial", 14, "bold")).pack(side="left", padx=10)
tk.Button(detect_send_frame, text="Send", command=send_images, bg="green", fg="white", font=("Arial", 14, "bold")).pack(side="left", padx=10)


# Right Frame: Camera Feed
camera_label = tk.Label(right_frame, bg=bg_color)
camera_label.pack(fill="both", expand=True)

# Camera Status
status_frame = tk.Frame(right_frame, bg=bg_color)
status_frame.place(relx=0.09, rely=0.86, anchor='sw')

# status_text_label = tk.Label(status_frame, text="Status:", font=("Arial", 14, "bold"), bg=bg_color, fg="white")
# status_text_label.pack(side="left")

camera_status_var = tk.StringVar(value="Not Recording")
camera_status_label = tk.Label(status_frame, textvariable=camera_status_var, font=("Arial", 14, "bold"), bg=bg_color, fg="green")
camera_status_label.pack(side="left", padx=5)

# Coordinates at the bottom-right
coordinates_var = tk.StringVar(value=f"Coordinates: {lat}, {lng}")

coord_frame = tk.Frame(right_frame, bg=bg_color)
coord_frame.pack(side="bottom", anchor="se", padx=10, pady=10, fill="x")

coordinates_label = ttk.Label(coord_frame, textvariable=coordinates_var, font=("Arial", 11), background=bg_color, foreground="white")
coordinates_label.pack(anchor="e")  # Aligns to bottom-right

def update_coordinates():
    coordinates_var.set(f"Coordinates: {lat}, {lng}")
    root.after(1, update_coordinates)  # as fast as Tkinter can handle

def post_gui_setup():
    global connErr
    if ready:
        # update_status("Ready to Run")
        restart_serial()
    else:
        if connErr == 1:
            update_status("MOTOR Arduino Disconnected")
        elif connErr == 2:
            update_status("GPS Arduino Disconnected")
        restart_serial()
 

# Start webcam feed
start_camera()
check_ready_state()
update_coordinates()

root.after(100, lambda: threading.Thread(target=record_loop, daemon=True).start())
root.after(100, post_gui_setup)
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()

