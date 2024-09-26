import h5py
import csv
import numpy as np


def calculate_num_objects(dataset, frame_key):
    """Calculate the number of objects in a frame."""
    return len(dataset[frame_key])


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


def calculate_mean_movement_magnitude(dataset, frame_key):
    """Calculate the average displacement of all objects between consecutive frames"""
    displacements = []
    for track_key in dataset[frame_key].keys():
        box = dataset[frame_key][track_key]["box"][:]
        if int(frame_key.split("_")[-1]) > 1:
            prev_frame_key = f"frame_{int(frame_key.split('_')[-1]) - 1:05d}"
            if prev_frame_key in dataset and track_key in dataset[prev_frame_key]:
                prev_box = dataset[prev_frame_key][track_key]["box"][:]
                dx = box[0] - prev_box[0]
                dy = box[1] - prev_box[1]
                displacement = np.sqrt(dx**2 + dy**2)
                displacements.append(displacement)
            else:
                displacements.append(0)
    mean_displacement = np.mean(displacements) if displacements else 0
    return mean_displacement

def calculate_movement_vector_angle_variance(dataset, frame_key):
    """Calculate the angular variance of object movements between frames"""
    angles = []
    for track_key in dataset[frame_key].keys():
        box = dataset[frame_key][track_key]["box"][:]
        if int(frame_key.split("_")[-1]) > 1:
            prev_frame_key = f"frame_{int(frame_key.split('_')[-1]) - 1:05d}"
            if prev_frame_key in dataset and track_key in dataset[prev_frame_key]:
                prev_box = dataset[prev_frame_key][track_key]["box"][:]
                dx = box[0] - prev_box[0]
                dy = box[1] - prev_box[1]
                angle = np.arctan2(dy, dx)
                angles.append(angle)
            else:
                angles.append(0)
    angle_variance = np.var(angles) if angles else 0
    return angle_variance



def calculate_interaction_count(boxes):
    """Count the number of object pairs per frame whose Euclidean distance is below a predefined proximity threshold"""
    proximity_threshold = 50  # pixels
    interaction_count = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            x1, y1, w1, h1 = boxes[i]
            x2, y2, w2, h2 = boxes[j]
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distance < proximity_threshold:
                interaction_count += 1
    return interaction_count

def calculate_mean_iou(boxes):
    """Calculate the Intersection over Union (IoU) for all object pairs per frame"""
    iou_values = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            x1, y1, w1, h1 = boxes[i]
            x2, y2, w2, h2 = boxes[j]
            intersection_area = calculate_intersection_area(x1, y1, w1, h1, x2, y2, w2, h2)
            union_area = w1 * h1 + w2 * h2 - intersection_area
            if union_area == 0:
                iou = 0
            else:
                iou = intersection_area / union_area
            iou_values.append(iou)
    if len(iou_values) == 0:
        return 0
    else:
        return np.mean(iou_values)

def calculate_spatial_density_ratio(boxes):
    """Compute the ratio of the total area occupied by all YOLO bounding boxes in a frame to the overall frame area"""
    frame_width = 640  # pixels
    frame_height = 480  # pixels
    total_area = 0
    for box in boxes:
        x, y, w, h = box
        total_area += w * h
    spatial_density_ratio = total_area / (frame_width * frame_height)
    return spatial_density_ratio

def calculate_intersection_area(x1, y1, w1, h1, x2, y2, w2, h2):
    """Calculate the intersection area of two rectangles"""
    intersection_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    intersection_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return intersection_x * intersection_y

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
                        "Mean Displacement",
                        "Angle Variance",
                        "Interaction Count",
                        "Mean IoU",
                        "Spatial Density",
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
                    mean_displacement = calculate_mean_movement_magnitude(
                        dataset, frame_key
                    )
                    angle_variance = calculate_movement_vector_angle_variance(
                        dataset, frame_key
                    )
                    interaction_count = calculate_interaction_count(boxes)
                    mean_iou = calculate_mean_iou(boxes)
                    spatial_density_ratio = calculate_spatial_density_ratio(boxes)

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
                        "mean_displacement": mean_displacement,
                        "angle_variance": angle_variance,
                        "interaction_count": interaction_count,
                        "mean_iou": mean_iou,
                        "spatial_density_ratio": spatial_density_ratio,
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
                            mean_displacement,
                            angle_variance,
                            interaction_count,
                            mean_iou,
                            spatial_density_ratio,
                        ]
                    )

        print(f"CSV file {csv_file_path} created successfully.")

    except Exception as e:
        # Handle exception
        print(f"Error occurred while calculating features: {e}")