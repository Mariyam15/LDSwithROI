MARIYAM FATIMA
https://github.com/Mariyam15/LDSwithROI


## Lane Detection System with ROI 

This project implements a lane detection pipeline in Python using OpenCV and NumPy.
It detects white and yellow lanes in images using the following steps:

### HSV color masking for lane colors (yellow & white)

### Gaussian blur for noise reduction

### Custom Canny edge detection

### Region of Interest (ROI) masking

### Hough line detection

# Lane line fitting and overlay

The project saves all intermediate steps and the final lane overlay for every input image.

## Folder Structure
lane_detection/
├─ main.py          # Main script, runs the batch processing
├─ lane_detector.py # LaneDetector class
├─ utils.py         # Helper functions
├─ README.md        # This file

## Installation

Clone or download the repository.

Install dependencies:

pip3 install opencv-python numpy

Usage

Run the pipeline from the terminal (VS Code terminal or any shell):

python3 lane_detection/main.py \
--input_folder "/full/path/to/input/images" \
--output_folder "/full/path/to/results_final"


--input_folder : Path to the folder containing input images.

--output_folder: Path where results will be saved. Subfolders for each processing step will be created automatically.

## Output

The output folder will have the following structure:

results_final/
├─ 2_blur/        # Gaussian blurred masks
├─ 3_edges/       # Custom Canny edges
├─ 4_roi_edges/   # Masked edges in ROI
├─ 5_hough/       # Hough lines overlay
├─ final/         # Final lane overlay


Each folder contains the corresponding processed images with the same filenames as the input.

Notes

The pipeline supports .jpg, .jpeg, and .png images.

You can enable or disable intermediate step outputs by changing the debug parameter in LaneDetector:

detector = LaneDetector(debug=True)  # True to save intermediate steps


Ensure your Python environment has OpenCV and NumPy installed.

Example
python3 lane_detection/main.py \
--input_folder "/home/user/images" \
--output_folder "/home/user/lane_results"


This will process all images in /home/user/images and save results in /home/user/lane_results.
