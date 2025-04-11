# import os
# import cv2 as cv
# import numpy as np
#
# def remove_audio(input_video_path, output_video_path):
#     cap = cv.VideoCapture(input_video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return
#
#     fps = cap.get(cv.CAP_PROP_FPS)
#     width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
#
#     fourcc = cv.VideoWriter_fourcc(*'mp4v')
#     out = cv.VideoWriter(output_video_path, fourcc, fps, (width, height))
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = np.rot90(frame, 3)
#
#         out.write(frame)
#
#     cap.release()
#     out.release()
#
#
# def main():
#     input_folder = "Data"
#     output_folder= "DataNoSound"
#
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     for filename in os.listdir(input_folder):
#         input_video_path = os.path.join(input_folder, filename)
#         output_video_path = os.path.join(output_folder, filename)
#         print(output_video_path)
#         print(input_video_path)
#         remove_audio(input_video_path, output_video_path)
import os
import cv2 as cv

def remove_audio(input_video_path, output_video_path):
    cap = cv.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    if total_frames == 0 or width == 0 or height == 0:
        print(f"Skipping {input_video_path} — unable to read metadata properly.")
        cap.release()
        return

    # Needed for rotating videos.
    frame_size = (height, width)

    # Use mp4v and ensure extension is .mp4
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_video_path, fourcc, fps, frame_size)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # needed for rotating videos
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    if frame_count == 0:
        print(f"Warning: No frames were written for {output_video_path}")
        if os.path.exists(output_video_path):
            os.remove(output_video_path)  # Remove empty file
    else:
        print(f"Saved {frame_count} frames to {output_video_path}")

def main():
    input_folder = "Data"
    output_folder = "DataNoSound"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".mov"):
            input_video_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".mp4"
            output_video_path = os.path.join(output_folder, output_filename)

            if os.path.exists(output_video_path):
                os.remove(output_video_path)  # Remove existing file
            print(f"Processing: {filename}")
            remove_audio(input_video_path, output_video_path)

if __name__ == "__main__":
    main()