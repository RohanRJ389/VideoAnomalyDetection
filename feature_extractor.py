import h5py
import csv
def calculate_num_objects(dataset, frame_key):
    """Calculate the number of objects in a frame"""
    return len(dataset[frame_key].keys())

def calculate_bounding_box_positions(dataset, frame_key):
    """Calculate the bounding box positions and areas for a frame"""
    boxes = []
    for track_key in dataset[frame_key].keys():
        box = dataset[frame_key][track_key]['box'][:]
        boxes.append(box)
    return boxes

def calculate_spatial_density(boxes):
    """Calculate the spatial density for a list of bounding boxes"""
    if len(boxes) > 0:
        areas = [box[2] * box[3] for box in boxes]
        total_area = sum(areas)
        return len(boxes) / total_area
    else:
        return 0


def calculate_features(h5_file_path, csv_file_path):
    """Calculate features for a given h5 file and save to a CSV file"""
    try:
        # Open the h5 file
        with h5py.File(h5_file_path, 'r') as h5file:
            print(f"Opened h5 file {h5_file_path}")
            # Get the dataset
            dataset = h5file['tracking_data']

            # Create the CSV file
            with open(csv_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write the header
                writer.writerow(['Frame Number', 'Number of Objects', 'Spatial Density'])

                # Create a dictionary to store the features per frame
                features_per_frame = {}

                # Iterate over the frames
                for frame_key in dataset.keys():
                    print(f"Processing frame {frame_key}")
                    num_objects = calculate_num_objects(dataset, frame_key)
                    print(f"Number of objects in frame {frame_key}: {num_objects}")
                    boxes = calculate_bounding_box_positions(dataset, frame_key)
                    print(f"Number of bounding boxes in frame {frame_key}: {len(boxes)}")
                    spatial_density = calculate_spatial_density(boxes)

                    # Store the features in the dictionary
                    frame_number = int(frame_key.split('_')[-1])
                    features_per_frame[frame_number] = {
                        'num_objects': num_objects,
                        'spatial_density': spatial_density
                    }

                    # Write the data to the CSV file
                    writer.writerow([frame_number, num_objects, spatial_density])

            print(f"CSV file {csv_file_path} created successfully.")

    except Exception as e:
        # Handle exception
        print(f"Error occurred while calculating features: {e}")