import cv2
import h5py
from ultralytics import YOLO
import os
from concurrent.futures import ThreadPoolExecutor
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def process_video_and_save_to_h5py(video_path, output_h5py_path, success_file_path, model_path='yolov8n.pt'):
    try:
        # Initialize YOLO model
        model = YOLO(model_path)

        # Open video file
        cap = cv2.VideoCapture(video_path)

        # Get total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        empty_frame_count = 0

        # Prepare to save tracking data
        with h5py.File(output_h5py_path, 'w') as h5file:
            dataset = h5file.create_group('tracking_data')

            frame_count = 0

            while cap.isOpened():
                # Read a frame from the video
                success, frame = cap.read()

                if not success:
                    break

                # Run YOLOv8 tracking on the frame using ByteTrack
                try:
                    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
                except Exception as e:
                    print(f"Error in tracking on frame {frame_count} in {video_path}: {e}")
                    continue  # Skip to the next frame in case of an error

                # Proceed only if results are valid and contain tracking data
                if results and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    confidences = results[0].boxes.conf.cpu().tolist()
                    class_ids = results[0].boxes.cls.int().cpu().tolist()

                    # Store features in h5py file
                    for i, (box, track_id, confidence, class_id) in enumerate(zip(boxes, track_ids, confidences, class_ids)):
                        track_data_group = dataset.create_group(f'frame_{frame_count:05}/track_{track_id}')
                        track_data_group.create_dataset('box', data=box)
                        track_data_group.create_dataset('confidence', data=confidence)
                        track_data_group.create_dataset('class_id', data=class_id)
                        track_data_group.create_dataset('track_id', data=track_id)
                else:
                    # Count this frame as empty if no results are returned
                    empty_frame_count += 1

                frame_count += 1

            # Release video capture object
            cap.release()

        # Log the successful processing
        with open(success_file_path, 'a') as f:
            f.write(f"{video_path}\n")

        return total_frames, empty_frame_count, frame_count, video_path

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None, None, None, video_path

def get_video_paths_from_folder(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mp4')]

def process_videos_in_parallel(video_paths, output_folder, success_file, failure_file, model_path='yolov5n.pt'):
    output_h5py_files = [os.path.join(output_folder, os.path.basename(video_path).replace('.mp4', '.h5')) for video_path in video_paths]

    successful_paths = []
    unsuccessful_paths = []

    with ThreadPoolExecutor(max_workers=1000) as executor:
        futures = [executor.submit(process_video_and_save_to_h5py, video_path, output_h5py_path, success_file)
                   for video_path, output_h5py_path in zip(video_paths, output_h5py_files)]

        for future in futures:
            result = future.result()
            if result[0] is not None:
                successful_paths.append(result[3])
            else:
                unsuccessful_paths.append(result[3])

    with open(success_file, 'w') as f:
        for path in successful_paths:
            f.write(f"{path}\n")

    with open(failure_file, 'w') as f:
        for path in unsuccessful_paths:
            f.write(f"{path}\n")

# if __name__ == '__main__':
#     video_folder = 'Videos'  # Change to your folder path
#     output_folder = r'Extracted YOLO features'
#     success_file = 'success.txt'
#     failure_file = 'failure.txt'

#     video_paths = get_video_paths_from_folder(video_folder)

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     process_videos_in_parallel(video_paths, output_folder, success_file, failure_file)