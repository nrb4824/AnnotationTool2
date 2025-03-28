import os
import cv2 as cv
import numpy as np

def remove_audio(input_video_path, output_video_path):
    cap = cv.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = np.rot90(frame, 3)

        out.write(frame)

    cap.release()
    out.release()

def main():
    input_folder = "Data"
    output_folder= "DataNoSound"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".mov"):
            input_video_path = os.path.join(input_folder, filename)
            output_video_path = os.path.join(output_folder, filename)
            print(output_video_path)
            print(input_video_path)
            remove_audio(input_video_path, output_video_path)

if __name__ == "__main__":
    main()