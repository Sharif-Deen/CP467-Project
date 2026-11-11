import os
import cv2

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(PROJECT_DIR, "Dataset")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "Processed Dataset")

# Dataset subfolders
CLASSES = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe"
]

# Image preprocessing steps for feature extraction
def preprocess_retina(image_path):

    # 1. load img
    image = cv2.imread(image_path)

    # 2. resize image
    image = cv2.resize(image, (512, 512))

    # 3. extract green channel
    green = image[:, :, 1]

    # 4. smooth the img using a Gaussian filter (adjust size?)
    blurred = cv2.GaussianBlur(
        green,
        (5, 5), 
        1.0
    )

    # 5. apply CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(blurred)

    return enhanced


# driver
def main():

    total_processed = 0

    print("Starting retinal image preprocessing...\n")

    # create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # start processing each severity folder
    for class_name in CLASSES:

        input_folder = os.path.join(
            DATASET_DIR, class_name
        )

        output_folder = os.path.join(
            OUTPUT_DIR, class_name
        )

        # create corresponding output folder
        os.makedirs(output_folder, exist_ok=True)

        print(f"Processing: {class_name}")
        
        # Iterate through each image in the current severity folder and preprocess each image 
        # output each img into its corresponding Severity folder in the new Dataset folder: "Processed Dataset"
        for filename in os.listdir(input_folder):

            input_path = os.path.join(
                input_folder, filename
            )

            output_path = os.path.join(
                output_folder, filename
            )

            # preprocess the image
            processed_image = preprocess_retina(
                input_path
            )

            # save the processed image
            cv2.imwrite(
                output_path,
                processed_image
            )

            total_processed += 1

        print(f"Finished: {class_name}\n")

    # summary
    print("-----------------------------------")
    print("Preprocessing complete!")
    print(f"Images processed: {total_processed}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-----------------------------------")


if __name__ == "__main__":
    main()
