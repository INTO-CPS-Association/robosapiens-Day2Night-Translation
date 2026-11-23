import cv2
import os
import yaml

def split_video_to_frames(video_path, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames to {output_dir}")


with open('./configurations/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

video_path = config['video_settings']['video_path']
out_path = config['dataset_settings']['images_path']

split_video_to_frames(video_path, out_path)

