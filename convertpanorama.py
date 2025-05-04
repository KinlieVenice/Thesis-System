import cv2
import numpy as np
import os

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

def process_videos_in_folder(folder_path):
    video_extensions = {'.avi', '.mp4', '.mov', '.mkv'}  # Supported formats
    output_folder = os.path.join(folder_path, "PANORAMA")
    os.makedirs(output_folder, exist_ok=True)
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in video_extensions:
            print(f"Processing video: {file_name}")
            frames = extract_frames(file_path)
            
            if len(frames) < 2:
                print("Not enough frames to stitch for", file_name)
                continue
            panorama = stitch_frames(frames)
            
            if panorama is not None:
                output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + "_panorama.jpg")
                cv2.imwrite(output_path, panorama)
                print(f"Panorama saved as {output_path}")
            else:
                print(f"Failed to create panorama for {file_name}")

if __name__ == "__main__":
    folder_path = r"D:\New folder"  # Change to your folder path
    process_videos_in_folder(folder_path)