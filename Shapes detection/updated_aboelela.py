import cv2
import os
import subprocess
import time
from rov_cam import RovCam
from ultralytics import YOLO


def capture_photo_in_one_function(rov_cam):
    """Captures a photo from the ROV camera when SPACE is pressed and streams live video."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    filename = f"captured_{int(time.time())}.jpg"
    image_path = os.path.join(script_dir, filename)

    print("Press 'SPACE' to capture a photo or 'ESC' to exit.")
    cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
    while True:
        frame = rov_cam.read()

        if frame is None or frame.size == 0:
            print("Error: Failed to capture image.")
            continue  # Keep retrying instead of exiting

        cv2.imshow("Camera Preview", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            print("Exiting without saving.")
            break
        elif key == 32:  # SPACE key
            cv2.imwrite(image_path, frame)
            print(f"Photo captured and saved at {image_path}.")
            break

    cv2.destroyAllWindows()
    return image_path



def run_yolo_and_count_shapes(image_path, model_path, base_output_dir, output_txt):
    """Runs YOLO detection and counts detected shapes."""
    os.makedirs(base_output_dir, exist_ok=True)
    project = os.path.join(base_output_dir, "yolo_results")
    os.makedirs(project, exist_ok=True)

    name = "detection_results"

    try:
        print("Running YOLO detection...")
        model = YOLO(model_path)
        results = model.predict(source=image_path, save=True, save_txt=True, project=project, name=name)
        print("YOLO detection completed successfully.")
    except Exception as e:
        print(f"Error during YOLO detection: {str(e)}")
        return

    results_dirs = [d for d in os.listdir(project) if d.startswith(name)]
    if not results_dirs:
        print("No results directory found.")
        return

    latest_dir = sorted(results_dirs, key=lambda x: os.path.getmtime(os.path.join(project, x)))[-1]
    label_dir = os.path.join(project, latest_dir, "labels")

    shape_counts = {}
    total_shapes = 0

    if not os.path.exists(label_dir):
        print(f"Label directory not found: {label_dir}")
        return

    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            with open(os.path.join(label_dir, label_file), "r") as f:
                for line in f:
                    class_id = int(line.split()[0])
                    shape_counts[class_id] = shape_counts.get(class_id, 0) + 1
                    total_shapes += 1

    with open(output_txt, "w") as txt_file:
        txt_file.write("Shape Detection Summary:\n")
        for class_id, count in shape_counts.items():
            txt_file.write(f"Class {class_id}: {count}\n")
        txt_file.write(f"Total Shapes: {total_shapes}\n")

    print(f"Results saved in {output_txt}")

    # Find the latest YOLO results directory
    results_dirs = [d for d in os.listdir(project) if d.startswith(name)]
    if not results_dirs:
        print("Error: No results directory found after YOLO detection.")
        return

    latest_dir = sorted(results_dirs, key=lambda x: os.path.getmtime(os.path.join(project, x)))[-1]
    label_dir = os.path.join(project, latest_dir, "labels")

    print(f"Looking for YOLO label files in: {label_dir}")

    shape_counts = {}
    total_shapes = 0

    if not os.path.exists(label_dir):
        print(f"Error: Label directory not found at {label_dir}. YOLO may not have detected anything.")
        return

    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    if not label_files:
        print("Warning: No label files found. No shapes were detected.")
        return

    for label_file in label_files:
        with open(os.path.join(label_dir, label_file), "r") as f:
            for line in f:
                class_id = int(line.split()[0])
                shape_counts[class_id] = shape_counts.get(class_id, 0) + 1
                total_shapes += 1

    if total_shapes == 0:
        print("Warning: No shapes detected in the image.")
        return

    print(f"Detected shapes: {shape_counts}")

    # Save shape detection results
    try:
        with open(output_txt, "w") as txt_file:
            txt_file.write("Shape Detection Summary:\n")
            for class_id, count in shape_counts.items():
                txt_file.write(f"Class {class_id}: {count}\n")
            txt_file.write(f"Total Shapes: {total_shapes}\n")
        print(f"Results saved successfully in {output_txt}")
    except Exception as e:
        print(f"Error saving shape counts file: {e}")


def calculate_points_from_results(txt_file, class_points, output_points_file):
    """Calculates points from detected shapes and saves the results."""
    if not os.path.exists(txt_file):
        print(f"Error: Expected results file missing - {txt_file}")
        return

    total_points = 0
    shape_counts = {}

    with open(txt_file, "r") as file:
        lines = file.readlines()
        if not lines:
            print("Warning: Shape counts file is empty!")
            return

        for line in lines:
            if line.startswith("Class"):
                parts = line.strip().split(":")
                class_id = int(parts[0].split()[1])
                count = int(parts[1].strip())
                shape_counts[class_id] = count

    with open(output_points_file, "w") as output_file:
        output_file.write("Shape Points Summary:\n")
        output_file.write("-----------------------\n")
        for class_id, count in shape_counts.items():
            points = class_points.get(class_id, 0) * count
            total_points += points
            output_file.write(f"Class {class_id}: {count} shapes, {points} points\n")
        output_file.write("-----------------------\n")
        output_file.write(f"Total Points: {total_points}\n")

    print(f"Total Points: {total_points}")
    print(f"Points calculation completed. Results saved in {output_points_file}")


if __name__ == "__main__":
    model_path = "Shapes detection/results_final/runs/detect/train/weights/best.pt"

    rov_cam = RovCam(5600)
    saved_image_path = capture_photo_in_one_function(rov_cam)

    if saved_image_path:
        base_output_dir = os.path.abspath(os.path.join(model_path, "../../../../../output"))
        shape_counts_file = os.path.join(base_output_dir, "shape_counts.txt")
        output_points_file = os.path.join(base_output_dir, "shape_points_summary.txt")
        class_points = {0: 20, 1: 5, 2: 15, 3: 10}

        print(f"Running YOLO detection on image: {saved_image_path}")
        run_yolo_and_count_shapes(saved_image_path, model_path, base_output_dir, shape_counts_file)

        print(f"Processing points calculation from: {shape_counts_file}")
        calculate_points_from_results(shape_counts_file, class_points, output_points_file)
    else:
        print("No saved image path provided. Please try again.")
