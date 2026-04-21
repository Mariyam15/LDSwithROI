import os
import argparse
import cv2
from lane_detector import LaneDetector
from utils import list_images, ensure_dir, read_image, save_image

def main(input_folder, output_folder):
    ensure_dir(output_folder)
    steps = ['2_blur', '3_edges', '4_roi_edges', '5_hough', 'final']
    for step in steps:
        ensure_dir(os.path.join(output_folder, step))

    image_paths = list_images(input_folder)
    if not image_paths:
        print(f"No images found in {input_folder}")
        return

    detector = LaneDetector(debug=True)

    for path in image_paths:
        filename = os.path.basename(path)
        img_bgr = read_image(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        final_img, debug_dict = detector.process_image(img_rgb)

        for step_name, img in debug_dict.items():
            save_image(os.path.join(output_folder, step_name, filename),
                       cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        save_image(os.path.join(output_folder, 'final', filename),
                   cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

        print(f"Processed {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', required=True, help='Path to input images')
    parser.add_argument('--output_folder', required=True, help='Path to save results')
    args = parser.parse_args()

    main(args.input_folder, args.output_folder)

