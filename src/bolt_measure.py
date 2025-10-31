import cv2
import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks
import warnings
import argparse

warnings.filterwarnings("ignore")


def measure_bolt(image_path):
    # --- 1. Load the Image ---
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"❌ Could not read the image file: {image_path}. Please check the path.")

    final_image = image.copy()

    # --- 2. Calculate pixels_per_mm from the Coin ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        image=gray_blurred,
        method=cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=50,
        maxRadius=250
    )

    if circles is None:
        raise SystemExit("❌ ERROR: No coin detected. Cannot calculate scale.")

    coin_circle = np.round(circles[0, :]).astype("int")[0]
    (coin_x, coin_y, coin_radius) = coin_circle
    KNOWN_COIN_DIAMETER_MM = 24.3
    pixels_per_mm = (coin_radius * 2) / KNOWN_COIN_DIAMETER_MM
    print(f"✅ Scale Calculated: {pixels_per_mm:.4f} pixels/mm")

    # --- 3. Isolate and Analyze the Bolt ---
    (h, w) = image.shape[:2]
    roi_start_x = int(w * 0.55)
    roi_end_x = int(w * 0.95)
    roi_start_y = int(h * 0.2)
    roi_end_y = int(h * 0.9)

    roi = gray[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
    edges = cv2.Canny(roi, 50, 150)

    # --- 4. Calculate Diameters ---
    points = np.argwhere(edges > 0)
    x_coords = points[:, 1]
    left_crest_x = int(np.percentile(x_coords, 5))
    right_crest_x = int(np.percentile(x_coords, 95))
    major_pixel_width = right_crest_x - left_crest_x
    major_diameter_mm = major_pixel_width / pixels_per_mm
    THREAD_DEPTH_MM = 0.812
    thread_depth_px = int(THREAD_DEPTH_MM * pixels_per_mm)
    left_root_x = left_crest_x + thread_depth_px
    right_root_x = right_crest_x - thread_depth_px
    minor_pixel_width = right_root_x - left_root_x
    minor_diameter_mm = minor_pixel_width / pixels_per_mm

    # --- 5. Pitch Calculation with Fine-Tuned Peak Finding ---
    vertical_projection = np.sum(edges, axis=1)

    min_peak_distance_px = int(1.0 * pixels_per_mm)
    peak_prominence = np.max(vertical_projection) * 0.25

    peaks, _ = find_peaks(
        vertical_projection,
        distance=min_peak_distance_px,
        prominence=peak_prominence
    )

    pitch_mm = 0
    if len(peaks) > 1:
        avg_peak_distance_px = np.mean(np.diff(peaks))
        pitch_mm = avg_peak_distance_px / pixels_per_mm
        for peak_y in peaks:
            cv2.line(
                final_image,
                (roi_start_x, roi_start_y + peak_y),
                (roi_end_x, roi_start_y + peak_y),
                (0, 165, 255),
                1
            )

    # --- 6. Draw Measurement Lines ---
    cv2.line(final_image, (roi_start_x + left_crest_x, roi_start_y),
             (roi_start_x + left_crest_x, roi_end_y), (0, 255, 0), 2)
    cv2.line(final_image, (roi_start_x + right_crest_x, roi_start_y),
             (roi_start_x + right_crest_x, roi_end_y), (0, 255, 0), 2)
    cv2.line(final_image, (roi_start_x + left_root_x, roi_start_y),
             (roi_start_x + left_root_x, roi_end_y), (255, 0, 0), 2)
    cv2.line(final_image, (roi_start_x + right_root_x, roi_start_y),
             (roi_start_x + right_root_x, roi_end_y), (255, 0, 0), 2)

    print("\n--- Complete Bolt Measurement Report ---")
    print(f"Major Diameter: {major_diameter_mm:.2f} mm")
    print(f"Minor Diameter: {minor_diameter_mm:.2f} mm")
    print(f"Pitch: {pitch_mm:.2f} mm")

    # --- 7. Check M10 Specifications ---
    major_dia_min = 9.2
    major_dia_max = 11.5

    is_major_dia_accepted = major_dia_min < major_diameter_mm < major_dia_max

    if is_major_dia_accepted:
        acceptance_status = "Accepted"
        print("✅ STATUS: Bolt measurements are within M10 tolerance.")
    else:
        acceptance_status = "Not Accepted"
        print("❌ STATUS: Bolt measurements are OUTSIDE M10 tolerance.")

    # --- 8. Save Image and Log Data ---
    output_dir = "results/output_images"
    os.makedirs(output_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_image_filename = os.path.join(output_dir, f"{base_filename}_measured.jpg")
    cv2.imwrite(output_image_filename, final_image)

    csv_filename = os.path.join("results", "bolt_measurements_log.csv")
    results_data = {
        'Input Image': [image_path],
        'Output Image': [output_image_filename],
        'Result': [acceptance_status],
        'Major Diameter (mm)': [round(major_diameter_mm, 2)],
        'Minor Diameter (mm)': [round(minor_diameter_mm, 2)],
        'Pitch (mm)': [round(pitch_mm, 2)],
        'Scale (pixels/mm)': [round(pixels_per_mm, 4)]
    }

    df = pd.DataFrame(results_data)
    df.to_csv(csv_filename, mode='a', header=not os.path.exists(csv_filename), index=False)
    print(f"✅ Results appended to {csv_filename}")
    print(f"✅ Output image saved at {output_image_filename}")

    # Optionally show image locally (uncomment if desired)
    # cv2.imshow("Final Image", final_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Bolt Measurement using Computer Vision")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    measure_bolt(args.image)
