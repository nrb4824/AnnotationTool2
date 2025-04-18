import os
import subprocess

def remove_audio(input_video_path, output_video_path):
    # ffmpeg command to copy video stream and remove audio
    cmd = [
        "ffmpeg",
        "-i", input_video_path,
        "-c:v", "copy",   # copy the video codec without re-encoding
        "-an",            # remove audio
        output_video_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"Processed: {output_video_path}")
    except subprocess.CalledProcessError:
        print(f"Error processing {input_video_path}")

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
                os.remove(output_video_path)

            remove_audio(input_video_path, output_video_path)

if __name__ == "__main__":
    main()
