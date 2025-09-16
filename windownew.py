import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import datetime
import os
from ultralytics import YOLO


class PCBApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PCB Inspection App")
        self.root.geometry("1000x750")

        # Load YOLO pretrained model (replace with your trained PCB model later)
        self.model = YOLO("yolov8n.pt")

        # Sidebar
        self.sidebar = tk.Frame(root, width=180, bg="#2c3e50")
        self.sidebar.pack(side="left", fill="y")

        # Main content area
        self.main_area = tk.Frame(root, bg="#ecf0f1")
        self.main_area.pack(side="right", expand=True, fill="both")

        # Sidebar buttons
        self.new_project_btn = tk.Button(
            self.sidebar, text="New Project (Camera)",
            bg="#34495e", fg="white", command=self.open_camera
        )
        self.new_project_btn.pack(pady=20, padx=10, fill="x")

        self.load_ref_btn = tk.Button(
            self.sidebar, text="Load Reference PCB",
            bg="#1abc9c", fg="white", command=self.load_reference
        )
        self.load_ref_btn.pack(pady=10, padx=10, fill="x")

        self.open_file_btn = tk.Button(
            self.sidebar, text="Open Test PCB (File)",
            bg="#16a085", fg="white", command=self.open_test_file
        )
        self.open_file_btn.pack(pady=10, padx=10, fill="x")

        # Initial label
        self.label = tk.Label(
            self.main_area,
            text="Welcome! Use 'New Project' to scan, 'Load Reference PCB' to set baseline, "
                 "and 'Open Test PCB' to check errors.",
            font=("Arial", 14),
            bg="#ecf0f1", wraplength=500, justify="center"
        )
        self.label.pack(pady=100)

        self.cap = None
        self.camera_running = False
        self.video_label = None
        self.capture_btn = None

        # Folders for saving
        self.save_folder = "Captured_PCB_Images"
        os.makedirs(self.save_folder, exist_ok=True)

        # Store reference detections
        self.reference_detections = None

    # -------------------- UTILS --------------------
    def get_detections(self, frame):
        """Run YOLO and return (label, box) list"""
        results = self.model(frame)
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            detections.append((label, xyxy))
        return detections, results[0].plot()

    def compare_pcbs(self, ref_detections, test_detections, test_img):
        """Compare reference vs test PCB and annotate errors"""
        annotated = test_img.copy()
        ref_labels = [d[0] for d in ref_detections]
        test_labels = [d[0] for d in test_detections]

        # Draw detected test components
        for label, box in test_detections:
            x1, y1, x2, y2 = box
            color = (0, 255, 0) if label in ref_labels else (0, 0, 255)  # green if correct, red if extra
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show missing components
        missing = [lbl for lbl in ref_labels if lbl not in test_labels]
        for i, lbl in enumerate(missing):
            cv2.putText(annotated, f"Missing: {lbl}", (20, 40 + 30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return annotated

    # -------------------- CAMERA MODE --------------------
    def open_camera(self):
        for widget in self.main_area.winfo_children():
            widget.destroy()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            return

        self.camera_running = True
        self.video_label = tk.Label(self.main_area)
        self.video_label.pack()

        self.capture_btn = tk.Button(self.main_area, text="Capture Image", command=self.capture_image)
        self.capture_btn.pack(pady=10)

        self.update_frame()

    def update_frame(self):
        if not self.camera_running:
            return

        ret, frame = self.cap.read()
        if ret:
            detections, annotated_frame = self.get_detections(frame)

            cv2image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.video_label.after(30, self.update_frame)

    def capture_image(self):
        if self.cap is None or not self.camera_running:
            messagebox.showwarning("Warning", "Camera is not running")
            return

        ret, frame = self.cap.read()
        if ret:
            detections, annotated_frame = self.get_detections(frame)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.save_folder}/pcb_{timestamp}.jpg"
            cv2.imwrite(filename, annotated_frame)
            messagebox.showinfo("Captured", f"Labeled image saved as {filename}")

    # -------------------- FILE MODE --------------------
    def load_reference(self):
        file_path = filedialog.askopenfilename(title="Select Reference PCB")
        if not file_path:
            return
        frame = cv2.imread(file_path)
        if frame is None:
            messagebox.showerror("Error", "Could not read reference image")
            return

        # Run YOLO detections
        self.reference_detections, annotated_frame = self.get_detections(frame)

        # Clear previous widgets
        for widget in self.main_area.winfo_children():
            widget.destroy()

        # Convert image for Tkinter
        cv2image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        # Show image
        lbl = tk.Label(self.main_area, image=imgtk)
        lbl.image = imgtk
        lbl.pack(pady=20)

        tk.Label(self.main_area,
                 text="Reference PCB loaded successfully!",
                 font=("Arial", 12), bg="#ecf0f1").pack(pady=10)

    def open_test_file(self):
        if self.reference_detections is None:
            messagebox.showwarning("Warning", "Load a Reference PCB first")
            return

        file_path = filedialog.askopenfilename(title="Select Test PCB")
        if not file_path:
            return
        frame = cv2.imread(file_path)
        if frame is None:
            messagebox.showerror("Error", "Could not read test image")
            return

        # Get detections
        test_detections, _ = self.get_detections(frame)

        # Compare with reference
        result_img = self.compare_pcbs(self.reference_detections, test_detections, frame)

        # Clear previous widgets
        for widget in self.main_area.winfo_children():
            widget.destroy()

        # Convert and show
        cv2image = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        lbl = tk.Label(self.main_area, image=imgtk)
        lbl.image = imgtk
        lbl.pack(pady=20)

        tk.Label(self.main_area,
                 text="Test PCB checked against reference!",
                 font=("Arial", 12), bg="#ecf0f1").pack(pady=10)

    # -------------------- EXIT --------------------
    def on_closing(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PCBApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
