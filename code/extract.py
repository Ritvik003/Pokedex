import cv2
import numpy as np
import os
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
import pandas as pd

def avg_color(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    avg_color = np.mean(image_rgb, axis=(0, 1))
    return avg_color

def spatial_grid_avg_colors(image_path, grid_size):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 4:
        a_channel = image[:, :, 3]
        mask = a_channel > 0  
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)  
    else:
        mask = np.ones((image.shape[0], image.shape[1]), dtype=bool) 
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, x = rgb.shape
    cell_h = height //grid_size[0]
    cell_w = width //grid_size[1]
    avg_colors = []

    for i in range(grid_size[0]):

        for j in range(grid_size[1]):
            cell_mask = mask[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            cell = rgb[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            cell = cell[cell_mask]
            if cell.size > 0:
                avg_color = np.mean(cell, axis=0)
            else:
                avg_color = np.array([0, 0, 0]) 
            avg_colors.append(avg_color)


    avg_colors_array = np.array(avg_colors)
    return avg_colors_array.reshape(-1, 3).flatten()

def color_histogram(image_path, bins=32):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        mask = (alpha > 0).astype(np.uint8) 
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        mask = None  


    histograms = []
    for i in range(3):  
        hist = cv2.calcHist([image],[i],mask,[bins],[0, 256])
        histograms.append(hist)
    histograms = np.concatenate(histograms).flatten()
    histograms = histograms / sum(histograms) 
    return histograms

def extract_hog_features(image_path):
    image = imread(image_path)
    image_gray = rgb2gray(image)
    features, hog_image = hog(image_gray, orientations=9, pixels_per_cell=(8, 8), 
                              cells_per_block=(2, 2), visualize=True, feature_vector=True)
    return features

def get_features(feat_type):
    db_path = "/Users/ritvikwarrier/Desktop/HW8-FinalProject/Data/Database"
    allimgs = [f for f in os.listdir(db_path) if f.endswith(('.png'))]
    features_list = []
    for img_file in allimgs:
        fullpath = os.path.join(db_path, img_file)
        if feat_type == "AVG_COLOR":
            features = avg_color(fullpath)
        elif feat_type == "SPATIAL":
            features = spatial_grid_avg_colors(fullpath, (4, 4))
        elif feat_type == "HIST":
            features = color_histogram(fullpath, 32)
        elif feat_type == "HOG":
            features = extract_hog_features(fullpath)
        else:
            raise ValueError("Unsupported feature type")

        features_list.append((img_file, features))

    feature_df = pd.DataFrame(features_list, columns=['Image', 'Features'])
    return feature_df



