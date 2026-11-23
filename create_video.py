import cv2
import os
from natsort import natsorted
import yaml

def synthesize_video_from_frames(frames_dir, output_video_path, fps=30):
    # Get all frame filenames
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))]
    frame_files = natsorted(frame_files)  # Sort numerically

    if not frame_files:
        raise ValueError("No frames found in the directory.")

    # Read the first frame to get the frame size
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 file
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Unable to read {frame_path}")
            continue
        out.write(frame)

    out.release()
    print(f"Video saved at {output_video_path}")

with open('./configurations/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

frames_path = config['results_settings']['translated_images_path']
out_path = config['video_settings']['video_export_path']
fps = int(config['video_settings']['video_fps'])

synthesize_video_from_frames(frames_path, out_path, fps=fps)

