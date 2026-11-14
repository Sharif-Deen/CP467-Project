import os
import pandas as pd

import cv2
import numpy as np

CLASS = {
    "No_DR": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Proliferate_DR": 4
}

FEATURE_COLUMNS = [
    "mean_intensity",
    "std_intensity",
    "p10_intensity",
    "p90_intensity",
    "entropy",
    "edge_density",
    "gradient_mean",
    "dark_response_mean",
    "bright_response_mean",
    "dark_candidate_fraction",
    "bright_candidate_fraction",
    "dark_small_spots_10k",
    "bright_small_spots_10k"
]


# Parameters
# These are just starting values they can be adjusted and experimented with.
IMAGE_SIZE = 512
BACKGROUND_THRESHOLD = 10
MASK_KERNEL_SIZE = 21
MORPH_KERNEL_SIZE = 25
CANDIDATE_THRESHOLD = 18
MIN_SPOT_AREA = 10
MAX_SPOT_AREA = 420

def get_retinal_mask(gray):
    """
    Each image contains a black background that surronds the retina
    This function returns a boolean mask, since the black background
    will get in the way of feature extraction
    """

    # Values near zero usually are apart of the black background.
    above_background = gray > BACKGROUND_THRESHOLD

    mask = above_background.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_KERNEL_SIZE, MASK_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    # Find desconnected white regions.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if count < 2:
        raise ValueError("Retinal region not found")

    # Choose the biggest foreground region, assuming its the retina.
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest).astype(np.uint8)


    #Keep only pixels belonging to the selected component.
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    retinal_mask = eroded_mask > 0

    # a quick error check, a 512 x 512 image should have thousands of pixels, so if its fewer then 500 something went wrong.
    if np.count_nonzero(retinal_mask) < 500:
        raise ValueError("Retinal mask to small")

    return retinal_mask


def count_small_components(binary_mask, retinal_pixel_count):
    """ Count small candidate regions per 10,000 retinal pixels."""

    # Connected components work on an 8bit binary image.
    binary_image = binary_mask.astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    small_spot_count = 0

    # Dont use label 0 since it represents the background. so start at 1.
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]

        if MIN_SPOT_AREA <= area <= MAX_SPOT_AREA:
            small_spot_count += 1

    # Normalize the count.
    spots_per_10k = small_spot_count * 10000.0 / retinal_pixel_count

    return float(spots_per_10k)

def extract_features(gray):
    '''Measure 13 features from a preprocessed 512x512 grayscale image.'''

    #Check to make sure images are in the correct format.
    if gray is None or gray.ndim != 2:
        raise ValueError("expected a grayscale image")

    if gray.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("expected 512x512 image")


    # Get only the retinal pixels.
    retinal_mask = get_retinal_mask(gray)
    retinal_pixels = gray[retinal_mask]
    retinal_pixel_count = retinal_pixels.size

    # Statistical Features
    # Intensity distribution and randomness
    mean_intensity = np.mean(retinal_pixels) / 255.0
    std_intensity = np.std(retinal_pixels) / 255.0
    p10_intensity = np.percentile(retinal_pixels, 10) / 255.0
    p90_intensity = np.percentile(retinal_pixels, 90) / 255.0

    # Entropy
    histogram = np.bincount(retinal_pixels, minlength=256)
    probabilities = histogram / retinal_pixel_count
    nonzero_probabilities = probabilities[probabilities > 0]
    entropy_terms = nonzero_probabilities * np.log2(nonzero_probabilities)
    entropy = -np.sum(entropy_terms)
    entropy = entropy / 8.0 # 8 bits is the max entropy for 256 values.

    # Edge features (Canny and sobel)
    # Canny finds sharp intensity transitions and counts edges inside of the retina.
    edges = cv2.Canny(gray, 50, 100)
    retinal_edges = edges[retinal_mask]
    edge_pixel_count = np.count_nonzero(retinal_edges)
    edge_density = edge_pixel_count / retinal_pixel_count

    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    gradient_mean = np.mean(gradient[retinal_mask])


    # Morphological Features.
    #Morphology highlights, locally dark/bright structures.
    # These features are candidates for microaneurysms, hemorrhages, or exudates.
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

    dark_response = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, morph_kernel)
    bright_response = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, morph_kernel)

    dark_response_mean = np.mean(dark_response[retinal_mask]) / 255.0
    bright_response_mean = np.mean(bright_response[retinal_mask]) / 255.0

    # Candiate Region features
    #Apply a threshold and ignore everything oustide the retina.
    dark_above_threshold = dark_response >= CANDIDATE_THRESHOLD
    bright_above_threshold = bright_response >= CANDIDATE_THRESHOLD

    dark_candidates = np.logical_and(dark_above_threshold, retinal_mask)
    bright_candidates = np.logical_and(bright_above_threshold, retinal_mask)

    # fraction of retinal region that is covered by the candidate pixels
    dark_candidate_pixels = np.count_nonzero(dark_candidates)
    bright_candidate_pixels = np.count_nonzero(bright_candidates)
    dark_candidate_fraction = dark_candidate_pixels / retinal_pixel_count
    bright_candidate_fraction = bright_candidate_pixels / retinal_pixel_count

    # Number of small connected candidate regions, normalized by the retinal area.
    dark_small_spots = count_small_components(dark_candidates, retinal_pixel_count)
    bright_small_spots = count_small_components(bright_candidates, retinal_pixel_count)


    # Return the named measurements of features.
    return {
        "mean_intensity": float(mean_intensity),
        "std_intensity": float(std_intensity),
        "p10_intensity": float(p10_intensity),
        "p90_intensity": float(p90_intensity),
        "entropy": float(entropy),
        "edge_density": float(edge_density),
        "gradient_mean": float(gradient_mean),
        "dark_response_mean": float(dark_response_mean),
        "bright_response_mean": float(bright_response_mean),
        "dark_candidate_fraction": float(dark_candidate_fraction),
        "bright_candidate_fraction": float(bright_candidate_fraction),
        "dark_small_spots_10k": float(dark_small_spots),
        "bright_small_spots_10k": float(bright_small_spots)
    }


def main(input_dir, output_csv):
    """ Read all five severity folders and write one feature row per image."""

    # Collect rows before writing the CSV file.
    rows = []
    for class_name, diagnosis in CLASS.items():
        class_folder = os.path.join(input_dir, class_name)

        if not os.path.isdir(class_folder):
            print("Missing folder:", class_folder)
            continue 


        print("Extracting features from:", class_name)

        for filename in sorted(os.listdir(class_folder)):
            valid_extension = filename.lower().endswith((".png", ".jpg", ".jpeg"))

            if not valid_extension:
                continue

            image_path = os.path.join(class_folder, filename)
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if gray is None:
                print("Could not read image:", image_path)
                continue

            try:
                features = extract_features(gray)
            except ValueError as error:
                print("Skipping", filename, "because", error)
                continue

            # Severity labels come from the the folder name the image is in.
            image_id = os.path.splitext(filename)[0]
            row = {
                "image_id":image_id,
                "class_name": class_name,
                "diagnosis": diagnosis
            }

            row.update(features)
            rows.append(row)

    if len(rows) == 0:
        raise RuntimeError("No usable images were found")

    # Create the output folder (Fatures/).
    output_folder = os.path.dirname(output_csv)
    os.makedirs(output_folder, exist_ok=True)

    # One image becomes a single CSV row: metadata + the same 13 feature columns
    column_names = ["image_id", "class_name", "diagnosis"]
    column_names.extend(FEATURE_COLUMNS)

    # convert the extracted features dictionary to a pandas table, then save the table as a CSV.
    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv(output_csv, index=False)

    print("Saved", len(rows), "images to", output_csv)


if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(code_dir)

    input_dir = os.path.join(project_dir, "Processed Dataset")
    output_csv = os.path.join(project_dir, "Features", "features.csv")

    main(input_dir, output_csv)


