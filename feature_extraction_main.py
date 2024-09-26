from all_features import *


def extract_features(h5_file_path):
    csv_file_path = 'features.csv'
    calculate_features(h5_file_path, csv_file_path)

# # Example usage
# h5_file_path = r'F:\VideoAnomalyDetection\Data\raw-yolo-data\subset_features\Anomaly\Abuse001_x264.h5'
# extract_features(h5_file_path)

import os
import shutil
from all_features import calculate_features

def extract_features_from_folder(h5_folder_path, csv_folder_path):
    """Extract features from all h5 files in a folder and save to csv files in another folder"""
    # Create the csv folder if it doesn't exist
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)

    # Get a list of all h5 files in the h5 folder
    h5_files = [f for f in os.listdir(h5_folder_path) if f.endswith('.h5')]

    # Extract features from each h5 file and save to a csv file
    for h5_file in h5_files:
        h5_file_path = os.path.join(h5_folder_path, h5_file)
        csv_file_path = os.path.join(csv_folder_path, h5_file.replace('.h5', '.csv'))
        calculate_features(h5_file_path, csv_file_path)
        print(f"Extracted features from {h5_file} and saved to {csv_file_path}")

# Example usage
h5_folder_path = r'F:\VideoAnomalyDetection\Data\raw-yolo-data\subset_features\Anomaly'
csv_folder_path = r'F:\VideoAnomalyDetection\Data\extracted-features\Anomaly'
extract_features_from_folder(h5_folder_path, csv_folder_path)


h5_folder_path = r'F:\VideoAnomalyDetection\Data\raw-yolo-data\subset_features\Anomaly'
csv_folder_path = r'F:\VideoAnomalyDetection\Data\extracted-features\Normal'
extract_features_from_folder(h5_folder_path, csv_folder_path)