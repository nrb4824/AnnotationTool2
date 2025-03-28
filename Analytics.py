import os
import json
import pandas as pd
import matplotlib.pyplot as plt


def count_unique_holds_used(tracking_data):
    unique_holds = set()
    for frame_data in tracking_data.values():
        for hold_id, hold_info in frame_data.items():
            if "frame_used" in hold_info:
                unique_holds.add(hold_id)

    return len(unique_holds)


def main():
    folder_path = "JsonOut" # Folder where the json files are stored

    data_list = []

    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    for filename in json_files:
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                tracking_data = data.get("tracking_data", {})

                start_frame = int(data.get("start_frame") or 0)
                end_frame = int(data.get("end_frame") or 0)
                fps = data.get("fps", 1)
                total_time = round((end_frame - start_frame) / fps) if fps > 0 and start_frame > 0 and end_frame > 0 else 0

                data_list.append({
                    "Total Frames": data.get("total_frames") or 0,
                    "Number of Holds": data.get("number_of_holds") or 0,
                    "Color of Route": data.get("color_of_route") or "Unknown",
                    "Route Rating": data.get("route_rating") or "Unknown",
                    "Completion": data.get("climb_valid") or "Unknown",
                    "Unique Holds Used": count_unique_holds_used(tracking_data),
                    "Time of Climb": total_time
                })
    df = pd.DataFrame(data_list)
    print(df)

    numeric_columns = ["Total Frames", "Number of Holds", "Unique Holds Used", "Time of Climb"]
    for column in numeric_columns:
        plt.figure()
        df[column].plot(kind='bar', figsize=(8, 6), legend=False)
        plt.title(column)
        plt.xlabel("Climbs")
        plt.ylabel(column)
        plt.savefig(f"Analytics/{column.replace(' ', '_').lower()}.png")
        plt.close()

    color_counts = df["Color of Route"].value_counts()
    plt.figure()
    color_counts.plot(kind='pie', autopct='%1.1f%%', colors=color_counts.index, legend=False)
    plt.title("Color of Route")
    plt.ylabel("")
    plt.savefig("Analytics/color_of_route.png")
    plt.close()

    # print(f"Number of JSON files: {len(json_files)}")
    # print(f"Total Frames: {total_frames_list}")
    # print(f"Number of Holds: {number_of_holds_list}")
    # print(f"Color of Routes: {color_of_route_list}")
    # print(f"Route Rating: {route_rating_list}")
    # print(f"Completion: {completetion_list}")
    # print(f"Unique Holds Used: {unique_holds_used}")
    # print(f"Time of Climb: {time_of_climb_list}")


if __name__ == "__main__":
    main()