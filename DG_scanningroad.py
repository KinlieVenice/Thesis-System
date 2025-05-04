import tkinter as tk
from tkinter import ttk, messagebox
import serial.tools.list_ports
import time
import cv2
from datetime import datetime
import threading

# Global variables for camera and frame sharing
cap = None
current_frame = None
frame_lock = threading.Lock()
running = True  # Flag to keep the live feed running

def live_feed():
    """Continuously update and display the live feed from the webcam."""
    global current_frame, running
    while running:
        ret, frame = cap.read()
        


        if not ret:
            continue
        with frame_lock:
            # Make a copy to be used for recording
            current_frame = frame.copy()

        height, width, _ = frame.shape

        # Define the Y-coordinate for the horizontal guide line (e.g., center)
        line_y = 325  

        # Draw a horizontal line across the frame (color: blue, thickness: 2)
        cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)
        cv2.imshow("Live Feed", frame)
        cv2.waitKey(1)

def get_com_ports():
    """Get available COM ports."""
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

def refresh_com_ports():
    """Refresh the COM port list."""
    com_ports = get_com_ports()
    com_port_dropdown['values'] = com_ports
    if com_ports:
        com_port_dropdown.current(0)
    else:
        messagebox.showinfo("Info", "No COM ports available.")

def record_video(ser):
    record = True
    while record:
        record_start = False
        while not record_start:
            if ser.in_waiting > 0:
                data = ser.readline().decode().strip()
                if data == "START RECORD":
                    record_start = True
                    
                    # Update the recording status label before starting recording
                    status_var.set("Recording")
                  

                if data == "END":
                    status_var.set("End of Multiple Segments Recording")
                    record = False
                    record_start = False
                    break

        else:    
            """
            Record video from the already active webcam until the serial device sends 'A'.
            The recording uses the global current_frame updated by the live feed thread.
            """
            timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
            filename = f"{timestamp}_record.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            # Get frame dimensions from the camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cropped_height = height - 65  # Crop bottom 80 pixels

            out = cv2.VideoWriter(filename, fourcc, 20.0, (width, cropped_height))
            
            

            while record_start:
                with frame_lock:
                    if current_frame is not None:
                        frame = current_frame.copy()
                    else:
                        continue
                # Crop the bottom 80 pixels
                cropped_frame = frame[:cropped_height, :]
                out.write(cropped_frame)

                # Check for serial input
                if ser.in_waiting > 0:
                    data = ser.readline().decode().strip()
                    if data == "STOP RECORD":
                        break  # Stop recording when "A" is received
                
                time.sleep(0.05)  # Approximate 20 fps recording rate

            out.release()
            
            # Use the main thread to update the status label
            root.after(0, lambda: status_var.set("Not Recording"))
            # messagebox.showinfo("Info", f"Video recording saved as {filename}")

    ser.close()

def run_program():
    """Run the program with the selected COM port and distance."""
    selected_port = com_port_dropdown.get()
    selected_distance = distance_var.get()

    if not selected_port:
        messagebox.showerror("Error", "Please select a COM port.")
        return

    if not selected_distance:
        messagebox.showerror("Error", "Please select a distance.")
        return

    try:
        ser = serial.Serial(selected_port, 9600, timeout=1)
        time.sleep(2)  # Wait for the serial connection to initialize
        ser.dtr = False  # Disable DTR to prevent reset
        ser.rts = False  # Disable RTS to prevent reset
        time.sleep(2)  # Additional delay for stabilization
        ser.write(f"{selected_distance}\n".encode())


        
        # Start recording video in a separate thread so the UI remains responsive
        threading.Thread(target=record_video, args=(ser,), daemon=True).start()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to communicate with Arduino: {e}")
        status_var.set("Not Recording")

def on_closing():   
    """Clean up the camera and close the program."""
    global running, cap
    running = False
    time.sleep(0.1)
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# Create the main window
root = tk.Tk()
root.title("Arduino COM Port Selector")
root.geometry("300x250")

# Distance dropdown
ttk.Label(root, text="Select No. of Segment:").pack(pady=5)
distance_var = tk.StringVar()
distance_dropdown = ttk.Combobox(root, textvariable=distance_var, state="readonly")
distance_dropdown['values'] = [str(i) for i in range(1, 11)]
distance_dropdown.pack(pady=5)
distance_dropdown.current(0)

# COM port dropdown
ttk.Label(root, text="Select COM Port:").pack(pady=5)
com_port_var = tk.StringVar()
com_port_dropdown = ttk.Combobox(root, textvariable=com_port_var, state="readonly")
com_port_dropdown.pack(pady=5)
refresh_com_ports()

# Refresh COM ports button
refresh_button = ttk.Button(root, text="Refresh COM Ports", command=refresh_com_ports)
refresh_button.pack(pady=5)

# Run button
run_button = ttk.Button(root, text="Run", command=run_program)
run_button.pack(pady=10)

# Recording status label
status_var = tk.StringVar(value="Not Recording")
status_label = ttk.Label(root, textvariable=status_var, foreground="red", font=("Helvetica", 12, "bold"))
status_label.pack(pady=10)

# Initialize the webcam and start the live feed thread
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    messagebox.showerror("Error", "Failed to open the webcam.")
else:
    threading.Thread(target=live_feed, daemon=True).start()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
