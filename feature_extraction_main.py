from feature_extractor import *
def extract_features(h5_file_path):
    csv_file_path = 'features.csv'
    calculate_features(h5_file_path, csv_file_path)

# Example usage
h5_file_path = r'F:\VideoAnomalyDetection\Data\raw-yolo-data\subset_features\Anomaly\Abuse001_x264.h5'
extract_features(h5_file_path)