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
    file_count = 0

    for filename in json_files:
        if filename.endswith(".json"):
            file_count += 1
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
    completed_climbs = df[df["Completion"] == "Complete"]
    non_completed_climbs = df[df["Completion"] != "Complete"]

    print(df)
    print("Number of unique Climber IDs:", df["Climber ID"].nunique())
    print("Number of unique Climb IDs:", df["Climb ID"].nunique())
    print("Number of files processed:", file_count)
    print("Number of climbs completed:", completed_climbs.shape[0])
    print("Number of climbs not completed:", non_completed_climbs.shape[0])

    percentages = (df["Unique Holds Used"] / df["Number of Holds"]) * 100
    # Compute average percentage
    average_percentage = percentages.mean()
    print(f"Average holds used per climb (as % of total holds): {average_percentage:.2f}%")

    average_time_completed = completed_climbs["Time of Climb"].mean()
    print(f"Average time of completed climbs: {average_time_completed:.2f} seconds")

    average_time_non_completed = non_completed_climbs["Time of Climb"].mean()
    print(f"Average time of completed climbs: {average_time_non_completed:.2f} seconds")

    numeric_columns = ["Number of Holds", "Unique Holds Used"]
    bar_color = "#7D55C7"
    for column in numeric_columns:
        plt.figure(figsize=(10, 6))

        data = df[column].dropna().astype(int)  # ensure integer values
        value_counts = data.value_counts().sort_index()  # get frequencies in order

        x = value_counts.index
        y = value_counts.values

        plt.bar(x, y, width=0.8, edgecolor='black', color=bar_color)  # width < 1 leaves space between bars
        plt.xticks(x)  # one tick per integer
        plt.title(column, fontsize=18)
        plt.xlabel(column, fontsize=18)
        plt.ylabel("Frequency", fontsize=18)
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

    # Group by Climb ID and compute average time, keeping the first route rating
    grouped = complete_df.groupby("Climb ID").agg({
        "Time of Climb": "mean",
        "Route Rating": "first"  # assuming route rating is consistent per climb
    })

    # Extract the data
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
    plt.boxplot(
        box_data,
        tick_labels=rating_order_filtered,
        patch_artist=True,
        boxprops=dict(facecolor=bar_color, edgecolor='black')
    )
    plt.title("Time of Climb vs Route Rating", fontsize=26, pad=20)
    plt.xlabel("Route Rating", fontsize=26, labelpad=20)
    plt.ylabel("Time of Climb", fontsize=26, labelpad=20)
    plt.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    plt.savefig("Analytics/time_vs_route_rating_complete.png")
    plt.close()
    #endregion

    #region Color pie_chart
    color_counts = df["Color of Route"].value_counts()
    plt.figure()
    color_counts.plot(kind='pie', autopct='%1.1f%%', colors=color_counts.index, legend=False)
    plt.title("Color of Route")
    plt.ylabel("")
    plt.savefig("Analytics/color_of_route.png")
    plt.close()
    #endregion

    #region Create the matrix
    climber_climb_matrix = df.pivot_table(
        index="Climber ID",
        columns="Climb ID",
        aggfunc="size",
        fill_value=0
    )

    # Convert to numpy array for plotting
    matrix_data = climber_climb_matrix.values

    # Set up the plot
    plt.figure(figsize=(12, 8))
    plt.imshow(matrix_data, cmap='Blues')

    # Set ticks
    plt.xticks(ticks=range(len(climber_climb_matrix.columns)), labels=climber_climb_matrix.columns, rotation=90)
    plt.yticks(ticks=range(len(climber_climb_matrix.index)), labels=climber_climb_matrix.index)

    # Labels and title
    plt.xlabel("Climb ID", fontsize=12)
    plt.ylabel("Climber ID", fontsize=12)
    plt.title("Climber vs Climb Matrix (Attempt Counts)", fontsize=16)

    # Show colorbar
    plt.colorbar(label='Number of Attempts')

    plt.tight_layout()
    plt.savefig("Analytics/climber_climb_matrix_visual.png")
    plt.close()
    #endregion

    #region Create the pivot table
    climber_rating_matrix = df.pivot_table(
        index="Route Rating",
        columns="Climber ID",
        aggfunc="size",
        fill_value=0
    )

    # Reorder the index to match the specified rating order
    climber_rating_matrix = climber_rating_matrix.reindex(index=rating_order, fill_value=0)

    # Convert to matrix for display
    matrix_data = climber_rating_matrix.values

    # Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(matrix_data, cmap='Purples', aspect='auto')

    # Set ticks with proper labels
    plt.xticks(ticks=range(len(climber_rating_matrix.columns)), labels=climber_rating_matrix.columns, fontsize=20)
    plt.yticks(ticks=range(len(climber_rating_matrix.index)), labels=climber_rating_matrix.index, fontsize=20)

    # Labels and title
    plt.xlabel("Climber ID", fontsize=26, labelpad=20)
    plt.ylabel("Route Rating", fontsize=26, labelpad=20)
    plt.title("Route Ratings vs Climbers", fontsize=26, pad=20)

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Number of Climbs', fontsize=26, labelpad=20)
    cbar.ax.tick_params(labelsize=20)

    plt.tight_layout()
    plt.savefig("Analytics/route_rating_vs_climber_matrix_visual.png")
    plt.close()
    #endregion


    #region Filter to only failed climbs
    failed_df = df[df["Completion"] == "Fell"]

    # Group by Climb ID and average the 'Unique Holds Used' and 'Time of Climb'
    grouped_failed = failed_df.groupby("Climb ID")[["Unique Holds Used", "Time of Climb"]].mean()

    # Extract averaged values
    x = grouped_failed["Unique Holds Used"]
    y = grouped_failed["Time of Climb"]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="#D9534F", edgecolors='black', alpha=0.7)

    plt.title("Avg Time vs Avg Unique Holds (Failed Climbs Only)", fontsize=16)
    plt.xlabel("Average Unique Holds Used", fontsize=14)
    plt.ylabel("Average Time of Climb (s)", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Analytics/avg_time_vs_avg_holds_failed_climbs.png")
    plt.close()
    #endregion


if __name__ == "__main__":
    main()