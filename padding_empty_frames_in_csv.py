import pandas as pd
def pad_empty_frames_in_csv(csv_file):
    df = pd.read_csv(csv_file)
    max_frame_number = df['Frame Number'].max()
    missing_frame_numbers = set(range(max_frame_number + 1)) - set(df['Frame Number'])
    for frame_number in missing_frame_numbers:
        zeros_str = ','.join(['0'] * 80)  # create a string of 80 comma-separated zeros
        new_row = [frame_number] + [0, 0, zeros_str] + [0] * (len(df.columns) - 4)  # adjust the number of zeros
        df.loc[len(df)] = new_row
    df = df.sort_values(by='Frame Number')  # sort the DataFrame by 'Frame Number'
    df.to_csv(csv_file, index=False)

import os

import os

def pad_empty_frames_in_folder(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            if os.path.getsize(file_path) == 0:
                # If the file is completely empty, add one row to it
                with open(file_path, 'w') as f:
                    f.write('Frame Number,Column1,Column2,Column3,...\n')  # Write the header row
                    zeros_str = ','.join(['0'] * 80)  # create a string of 80 comma-separated zeros
                    f.write('0,0,0,{},...\n'.format(zeros_str))  # Write one row with zeros
                pad_empty_frames_in_csv(file_path)  # Call pad_empty_frames_in_csv on the file
                print(file_name,'done')
            else:
                # If the file is not empty, call pad_empty_frames_in_csv on the file
                pad_empty_frames_in_csv(file_path)
                print(file_name,'was empty, done')                
pad_empty_frames_in_folder(r"C:\My_Stuff\College\Capstone Project\CAPSTONE\Capstone_2\VideoAnomalyDetection\extracted_features_csv\anomalous")
pad_empty_frames_in_folder(r"C:\My_Stuff\College\Capstone Project\CAPSTONE\Capstone_2\VideoAnomalyDetection\extracted_features_csv\normal")