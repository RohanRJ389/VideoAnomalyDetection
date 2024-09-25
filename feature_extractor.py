import h5py
import csv

def calculate_features(h5_file_path, csv_file_path):
    try:
        # Open the h5 file
        with h5py.File(h5_file_path, 'r') as h5file:
            # Get the dataset
            dataset = h5file['tracking_data']

            # Create a dictionary to store the features per frame
            features_per_frame = {}

            # Iterate over the frames
            for frame_key in dataset.keys():
                # Get the number of objects in the frame
                num_objects = len(dataset[frame_key].keys())

                # Get the bounding box positions and areas
                boxes = []
                for track_key in dataset[frame_key].keys():
                    box = dataset[frame_key][track_key]['box'][:]
                    boxes.append(box)

                # Calculate the spatial density
                if len(boxes) > 0:
                    areas = [box[2] * box[3] for box in boxes]
                    total_area = sum(areas)
                    spatial_density = len(boxes) / total_area
                else:
                    spatial_density = 0

                # Store the features in the dictionary
                frame_number = int(frame_key.split('_')[-1])
                features_per_frame[frame_number] = {
                    'num_objects': num_objects,
                    'spatial_density': spatial_density
                }

        # Create the CSV file
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header
            writer.writerow(['Frame Number', 'Number of Objects', 'Spatial Density'])

            # Write the data
            for frame_number, features in features_per_frame.items():
                writer.writerow([frame_number, features['num_objects'], features['spatial_density']])

        print(f"CSV file {csv_file_path} created successfully.")

    except Exception as e:
        print(f"Error processing {h5_file_path}: {e}")

def extract_features(h5_file_path):
    csv_file_path = 'features.csv'
    calculate_features(h5_file_path, csv_file_path)

# Example usage
h5_file_path = r'F:\VideoAnomalyDetection\Data\raw-yolo-data\subset_features\Anomaly\Abuse001_x264.h5'
extract_features(h5_file_path)