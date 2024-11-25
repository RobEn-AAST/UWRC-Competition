import os
import cv2
import subprocess


def run_yolo_and_count_shapes(image_path: str, model_path: str, output_dir: str, output_txt: str):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set up YOLO project and name arguments
    project = output_dir
    name = "detection_results"

    # YOLO detection command
    command = [
        "yolo", "task=detect", "mode=predict",
        f"model={model_path}", f"source={image_path}",
        f"project={project}", f"name={name}", "save=True", "save_txt=True"
    ]

    # Run YOLO
    try:
        subprocess.run(command, check=True)
        print("YOLO detection completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error during YOLO detection: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

    # Find the latest results directory
    results_dirs = [d for d in os.listdir(output_dir) if d.startswith(name)]
    if not results_dirs:
        print("No results directory found.")
        return

    # Sort directories to get the latest one
    latest_dir = sorted(results_dirs, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))[-1]
    label_dir = os.path.join(output_dir, latest_dir, "labels")

    # Count detected shapes
    shape_counts = {}
    total_shapes = 0

    if not os.path.exists(label_dir):
        print(f"Label directory not found: {label_dir}")
        return

    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            with open(os.path.join(label_dir, label_file), "r") as f:
                for line in f:
                    class_id = int(line.split()[0])  # Get the class ID from the label file
                    shape_counts[class_id] = shape_counts.get(class_id, 0) + 1
                    total_shapes += 1

    # Save counts to a text file
    with open(output_txt, "w") as txt_file:
        txt_file.write("Shape Detection Summary:\n")
        for class_id, count in shape_counts.items():
            txt_file.write(f"Class {class_id}: {count}\n")
        txt_file.write(f"Total Shapes: {total_shapes}\n")

    print(f"Results saved in {output_txt}")

def calculate_points_from_results(txt_file: str, class_points: dict, output_points_file: str):
    # Ensure the input file exists
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Results file not found: {txt_file}")

    # Initialize total points
    total_points = 0

    # Parse the results from the text file
    shape_counts = {}

    # Check if the file has content
    if os.stat(txt_file).st_size == 0:
        print("Warning: shape_counts.txt is empty!")
        return

    print(f"Reading results from: {txt_file}")

    with open(txt_file, "r") as file:
        for line in file:
            if line.startswith("Class"):
                parts = line.strip().split(":")
                class_id = int(parts[0].split()[1])  # Extract class ID
                count = int(parts[1].strip())  # Extract count
                shape_counts[class_id] = count

    print(f"Shape counts: {shape_counts}")  # Debugging output

    # Write the points summary to the output file
    with open(output_points_file, "w") as output_file:
        output_file.write("Shape Points Summary:\n")
        output_file.write("-----------------------\n")
        for class_id, count in shape_counts.items():
            points = class_points.get(class_id, 0) * count  # Default points for unknown classes is 0
            total_points += points
            output_file.write(f"Class {class_id}: {count} shapes, {points} points\n")
        output_file.write("-----------------------\n")
        output_file.write(f"Total Points: {total_points}\n")

    print(f"Total Points: {total_points}")
    print(f"Points calculation completed. Results saved in {output_points_file}")
    
    

def capture_image_from_camera(output_image_path: str):
    # Open the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return False

    print("Press 'c' to capture an image or 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read from the camera.")
            break

        # Display the frame
        cv2.imshow("Camera Feed", frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # Capture the image when 'c' is pressed
            cv2.imwrite(output_image_path, frame)
            print(f"Image captured and saved to {output_image_path}")
            break
        elif key == ord('q'):  # Quit when 'q' is pressed
            print("Exiting without capturing.")
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

    return True


# Example usage
if __name__ == "__main__":
    # Paths and parameters
    model_path = "C:/Users/HP/OneDrive/Desktop/results_final/runs/detect/train/weights/best.pt"
    output_dir = "C:/Users/HP/OneDrive/Desktop/results_finaloutput"
    shape_counts_file = "C:/Users/HP/OneDrive/Desktop/shape_counts.txt"
    output_points_file = "C:/Users/HP/OneDrive/Desktop/shape_points_summary.txt"
    class_points = {0: 20, 1: 5, 2: 15, 3: 10}  # Points for each class
    captured_image_path = "C:/Users/HP/OneDrive/Desktop/results_final/captured_image.jpg"

    # Step 1: Capture image from the camera
    if capture_image_from_camera(captured_image_path):
        # Step 2: Run YOLO and save shape counts
        run_yolo_and_count_shapes(captured_image_path, model_path, output_dir, shape_counts_file)

        # Step 3: Calculate points based on shape counts
        calculate_points_from_results(shape_counts_file, class_points, output_points_file)

