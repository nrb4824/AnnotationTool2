import os
import json
import pandas as pd
import numpy as np
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
                    "Climb ID": data.get("climb_id") or 0,
                    "Climber ID": data.get("climber_id") or 0,
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

    numeric_columns = ["Number of Holds", "Unique Holds Used"]
    for column in numeric_columns:
        plt.figure(figsize=(8, 6))

        data = df[column].dropna().astype(int)  # ensure integer values
        value_counts = data.value_counts().sort_index()  # get frequencies in order

        x = value_counts.index
        y = value_counts.values

        plt.bar(x, y, width=0.8, edgecolor='black')  # width < 1 leaves space between bars
        plt.xticks(x)  # one tick per integer
        plt.title(column)
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"Analytics/{column.replace(' ', '_').lower()}.png")
        plt.close()


    # region: Section holds vs holds used
    x_col = "Number of Holds"
    y_col = "Unique Holds Used"

    # Filter to only complete climbs
    complete_df = df[df["Completion"] == "Complete"]

    x = complete_df[x_col].dropna()
    y = complete_df[y_col].dropna()

    # Align x and y (in case of missing values)
    valid = x.index.intersection(y.index)
    x = x.loc[valid]
    y = y.loc[valid]

    # Fit a 2nd-degree polynomial (quadratic)
    coeffs = np.polyfit(x, y, deg=2)
    poly = np.poly1d(coeffs)

    # Generate smooth curve for the best fit
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = poly(x_fit)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7, label='Complete Climbs')
    plt.plot(x_fit, y_fit, color='red', label=f'Best Fit: {poly}')
    plt.title(f"{y_col} vs {x_col} (Complete Climbs Only)")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Analytics/holds_vs_unique_holds_complete.png")
    plt.close()
    # endregion

    #region Section time vs grade complete only
    # Filter to only complete climbs (where 'Completion' status is marked as 'Complete')
    complete_df = df[df["Completion"] == "Complete"]

    # Group by Climb ID and get the last entry (you can adjust this as needed)
    grouped = complete_df.groupby("Climb ID").last()

    # Make sure 'Route Rating' and 'Time of Climb' exist
    x = grouped["Route Rating"]
    y = grouped["Time of Climb"]

    # Define the desired order of ratings
    rating_order = ["Beg", "Nov", "Int", "Adv", "Exp"]

    # Filter out ratings not in the desired order
    rating_order_filtered = [rating for rating in rating_order if rating in x.unique()]

    # Prepare data for boxplot
    box_data = [y[x == rating] for rating in rating_order_filtered]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(box_data, labels=rating_order_filtered, patch_artist=True)
    plt.title("Time of Climb vs Route Rating (Complete Climbs Only)")
    plt.xlabel("Route Rating")
    plt.ylabel("Time of Climb")
    plt.tight_layout()
    plt.savefig("Analytics/time_vs_route_rating_complete.png")
    plt.close()
    #endregion

    color_counts = df["Color of Route"].value_counts()
    plt.figure()
    color_counts.plot(kind='pie', autopct='%1.1f%%', colors=color_counts.index, legend=False)
    plt.title("Color of Route")
    plt.ylabel("")
    plt.savefig("Analytics/color_of_route.png")
    plt.close()


if __name__ == "__main__":
    main()