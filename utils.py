import os
import cv2

def list_images(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def ensure_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def read_image(path):
    return cv2.imread(path)

def save_image(path, img):
    cv2.imwrite(path, img)
