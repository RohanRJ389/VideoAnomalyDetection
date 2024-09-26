import h5py
import csv
import numpy as np


def calculate_num_objects(dataset, frame_key):
    """Calculate the number of objects in a frame"""
    return len(dataset[frame_key].keys())


def calculate_bounding_box_positions(dataset, frame_key):
    """Calculate the bounding box positions and areas for a frame"""
    boxes = []
    for track_key in dataset[frame_key].keys():
        box = dataset[frame_key][track_key]["box"][:]
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


def calculate_class_distribution(dataset, frame_key):
    """Calculate the class-wise count of objects within the frame"""
    class_counts = np.zeros(80)  # Initialize a fixed-size 1D vector of length 80
    for track_key in dataset[frame_key].keys():
        class_id = dataset[frame_key][track_key]["class_id"]
        class_counts[class_id] += 1
    total_objects = np.sum(class_counts)
    if total_objects > 0:
        class_distribution = class_counts / total_objects  # Normalize into percentages
    else:
        class_distribution = class_counts
    return class_distribution

def calculate_velocity(dataset, frame_key):
    """Calculate velocity for each object in a frame"""
    velocities = []
    for track_key in dataset[frame_key].keys():
        box = dataset[frame_key][track_key]["box"][:]
        if int(frame_key.split("_")[-1]) > 1:
            prev_frame_key = f"frame_{int(frame_key.split('_')[-1]) - 1:05d}"
            if prev_frame_key in dataset and track_key in dataset[prev_frame_key]:
                prev_box = dataset[prev_frame_key][track_key]["box"][:]
                dx = box[0] - prev_box[0]
                dy = box[1] - prev_box[1]
                velocity = np.sqrt(dx**2 + dy**2)
                velocities.append(velocity)
            else:
                velocities.append(0)
    mean_velocity = np.mean(velocities) if velocities else 0
    max_velocity = np.max(velocities) if velocities else 0
    var_velocity = np.var(velocities) if velocities else 0
    return mean_velocity, max_velocity, var_velocity

def calculate_acceleration(dataset, frame_key):
    """Calculate acceleration for each object in a frame"""
    accelerations = []
    for track_key in dataset[frame_key].keys():
        box = dataset[frame_key][track_key]["box"][:]
        if int(frame_key.split("_")[-1]) > 2:
            prev_frame_key = f"frame_{int(frame_key.split('_')[-1]) - 1:05d}"
            prev_prev_frame_key = f"frame_{int(frame_key.split('_')[-1]) - 2:05d}"
            if (
                prev_frame_key in dataset
                and prev_prev_frame_key in dataset
                and track_key in dataset[prev_frame_key]
                and track_key in dataset[prev_prev_frame_key]
            ):
                prev_box = dataset[prev_frame_key][track_key]["box"][:]
                prev_prev_box = dataset[prev_prev_frame_key][track_key]["box"][:]
                dx = box[0] - prev_box[0]
                dy = box[1] - prev_box[1]
                velocity = np.sqrt(dx**2 + dy**2)
                prev_dx = prev_box[0] - prev_prev_box[0]
                prev_dy = prev_box[1] - prev_prev_box[1]
                prev_velocity = np.sqrt(prev_dx**2 + prev_dy**2)
                acceleration = velocity - prev_velocity
                accelerations.append(acceleration)
            else:
                accelerations.append(0)
    mean_acceleration = np.mean(accelerations) if accelerations else 0
    max_acceleration = np.max(accelerations) if accelerations else 0
    var_acceleration = np.var(accelerations) if accelerations else 0
    return mean_acceleration, max_acceleration, var_acceleration


import math

def calculate_directionality(dataset, frame_key):
    """Calculate directionality for each object in a frame"""
    directions = []
    for track_key in dataset[frame_key].keys():
        box = dataset[frame_key][track_key]["box"][:]
        if int(frame_key.split("_")[-1]) > 1:
            prev_frame_key = f"frame_{int(frame_key.split('_')[-1]) - 1:05d}"
            if prev_frame_key in dataset and track_key in dataset[prev_frame_key]:
                prev_box = dataset[prev_frame_key][track_key]["box"][:]
                dx = box[0] - prev_box[0]
                dy = box[1] - prev_box[1]
                direction = math.atan2(dy, dx)
                directions.append(direction)
            else:
                directions.append(0)
    mean_direction = np.mean(directions) if directions else 0
    direction_variance = np.var(directions) if directions else 0
    return mean_direction, direction_variance

def calculate_features(h5_file_path, csv_file_path):
    """Calculate features for a given h5 file and save to a CSV file"""
    try:
        # Open the h5 file
        with h5py.File(h5_file_path, "r") as h5file:
            print(f"Opened h5 file {h5_file_path}")
            # Get the dataset
            dataset = h5file["tracking_data"]

            # Create the CSV file
            with open(csv_file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                # Write the header
                writer.writerow(
                    [
                        "Frame Number",
                        "Number of Objects",
                        "Spatial Density",
                        "Class Distribution",
                        "Mean Velocity",
                        "Max Velocity",
                        "Variance in Velocity",
                        "Mean Acceleration",
                        "Max Acceleration",
                        "Variance in Acceleration",
                        "Mean Direction",
                        "Direction Variance",
                    ]
                )

                # Create a dictionary to store the features per frame
                features_per_frame = {}

                # Iterate over the frames
                for frame_key in dataset.keys():
                    print(f"Processing frame {frame_key}")
                    num_objects = calculate_num_objects(dataset, frame_key)
                    print(f"Number of objects in frame {frame_key}: {num_objects}")
                    boxes = calculate_bounding_box_positions(dataset, frame_key)
                    print(
                        f"Number of bounding boxes in frame {frame_key}: {len(boxes)}"
                    )
                    spatial_density = calculate_spatial_density(boxes)
                    class_distribution = calculate_class_distribution(
                        dataset, frame_key
                    )
                    mean_velocity, max_velocity, var_velocity = calculate_velocity(
                        dataset, frame_key
                    )
                    mean_acceleration, max_acceleration, var_acceleration = calculate_acceleration(
                        dataset, frame_key
                    )
                    mean_direction, direction_variance = calculate_directionality(
                        dataset, frame_key
                    )

                    # Store the features in the dictionary
                    frame_number = int(frame_key.split("_")[-1])
                    features_per_frame[frame_number] = {
                        "num_objects": num_objects,
                        "spatial_density": spatial_density,
                        "class_distribution": class_distribution,
                        "mean_velocity": mean_velocity,
                        "max_velocity": max_velocity,
                        "var_velocity": var_velocity,
                        "mean_acceleration": mean_acceleration,
                        "max_acceleration": max_acceleration,
                        "var_acceleration": var_acceleration,
                        "mean_direction": mean_direction,
                        "direction_variance": direction_variance,
                    }

                    # Write the data to the CSV file
                    writer.writerow(
                        [
                            frame_number,
                            num_objects,
                            spatial_density,
                            ",".join(map(str, class_distribution)),
                            mean_velocity,
                            max_velocity,
                            var_velocity,
                            mean_acceleration,
                            max_acceleration,
                            var_acceleration,
                            mean_direction,
                            direction_variance,
                        ]
                    )

        print(f"CSV file {csv_file_path} created successfully.")

    except Exception as e:
        # Handle exception
        print(f"Error occurred while calculating features: {e}")