import cv2
import os
import subprocess

def capture_photo_in_one_function():
    # Get the directory of the running script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate a unique filename based on a counter or timestamp
    filename = "captured_photo.jpg"  # Change this to a dynamic name if needed
    image_path = os.path.join(script_dir, filename)  # Save in the same folder as the script

    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return None  # No photo captured

    print("Press 'SPACE' to capture a photo or 'ESC' to exit.")
    indicator = 0
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display live camera feed
        cv2.imshow("Camera Preview", frame)

        # Listen for key press
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            print("Exiting without saving.")
            image_path = None  # No photo saved
            break
        elif key == 32:  # SPACE key
            # Save the image
            cv2.imwrite(image_path, frame)
            print(f"Photo captured and saved at {image_path}.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Return the image path (or None if no photo was captured)
    return image_path

# Example usage:
saved_image_path = capture_photo_in_one_function()

def run_yolo_and_count_shapes(image_path: str, model_path: str, base_output_dir: str, output_txt: str):
    # Create the base output directory if it doesn't exist
    os.makedirs(base_output_dir, exist_ok=True)

    # Define the YOLO project directory under the base output folder
    project = os.path.join(base_output_dir, "yolo_results")
    os.makedirs(project, exist_ok=True)

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
    results_dirs = [d for d in os.listdir(project) if d.startswith(name)]
    if not results_dirs:
        print("No results directory found.")
        return

    # Sort directories to get the latest one
    latest_dir = sorted(results_dirs, key=lambda x: os.path.getmtime(os.path.join(project, x)))[-1]
    label_dir = os.path.join(project, latest_dir, "labels")

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


# Example usage
if __name__ == "__main__":
    # Paths and parameters
    if(saved_image_path):
        image_path = saved_image_path
        model_path = "C:\\Users\\pc\\Desktop\\results_final\\runs\\detect\\train\\weights\\best.pt"

        # Set output_dir to "output" folder
        base_output_dir = os.path.abspath(os.path.join(model_path, "../../../../../output"))

        # Define the paths for saving results
        shape_counts_file = os.path.join(base_output_dir, "shape_counts.txt")
        output_points_file = os.path.join(base_output_dir, "shape_points_summary.txt")
        class_points = {0: 20, 1: 5, 2: 15, 3: 10}  # Points for each class

        # Step 1: Run YOLO and save shape counts
        run_yolo_and_count_shapes(image_path, model_path, base_output_dir, shape_counts_file)

        # Step 2: Calculate points based on shape counts
        calculate_points_from_results(shape_counts_file, class_points, output_points_file)
    else:
        print("No saved image path provided. Please provide a saved image path to run the script.")