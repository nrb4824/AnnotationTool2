import cv2 as cv
import numpy as np
import time
import json
import os
from datetime import datetime

class TemplateTracker:
    def __init__(self, template, location):
        self.template = template
        self.last_location = location
        self.tracking = True
        self.hold_used = False
        self.hold_used_frame = None
        self.start_hold = False
        self.end_hold = False

    def get_search_region(self, frame_shape, search_radius):
        if self.last_location is not None:
            center_x = self.last_location[0] + self.template.shape[1] // 2
            center_y = self.last_location[1] + self.template.shape[0] // 2

            start_x = max(0, center_x - self.template.shape[1] // 2 - search_radius)
            start_y = max(0, center_y - self.template.shape[0] // 2 - search_radius)
            end_x = min(frame_shape[1], center_x + self.template.shape[1] // 2 + search_radius)
            end_y = min(frame_shape[0], center_y + self.template.shape[0] // 2 + search_radius)

            search_region = (start_x, start_y, end_x - start_x, end_y - start_y)
        else:
            search_region = (0, 0, frame_shape[1], frame_shape[0])
        return search_region

    def update_location(self, frame, confidence_threshold, search_radius):
        search_region = self.get_search_region(frame.shape, search_radius)
        loc = find_template(frame, self.template, confidence_threshold, search_region)
        if loc is not None:
            self.last_location = loc
            self.tracking = True
        else:
            self.tracking = False
            loc = find_template(frame, self.template, confidence_threshold, search_region)
        return search_region, loc
    def set_hold_used(self, frame):
        if self.hold_used:
            self.hold_used = False
            self.hold_used_frame = None
        else:
            self.hold_used = True
            self.hold_used_frame = frame

    def get_hold_used(self):
        if self.hold_used:
            return self.hold_used_frame, self.hold_used
        else:
            return None, None

    def set_start_hold(self):
        self.start_hold = False if self.start_hold else True

    def get_start_hold(self):
        return self.start_hold

    def set_end_hold(self):
        self.end_hold = False if self.end_hold else True

    def get_end_hold(self):
        return self.end_hold


def load_all_frames(vidfile: str, scale_factor):
    vid = cv.VideoCapture(vidfile)
    frames = []
    fps = vid.get(cv.CAP_PROP_FPS)
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame =np.rot90(frame, 3)
        frame = cv.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor) # Resize frames for faster loading
        frames.append(frame)
    height, width, _ = frames[0].shape
    return frames, (width, height), fps

def extract_template(frame, bbox_topleft, bbox_bottomright):
    left, top = bbox_topleft
    right, bottom = bbox_bottomright
    return frame[top:bottom, left:right]

def find_template(frame, template, confidence_threshold, search_region, scale=1.0):
    x, y, w, h = search_region
    search_area = frame[y:y+h, x:x+w] # Crop the search region
    search_area, template = cv.resize(search_area, (0, 0), fx=scale, fy=scale), cv.resize(template, (0, 0), fx=scale,
                                                                                          fy=scale)
    res = cv.matchTemplate(search_area, template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    if max_val < confidence_threshold:
        return None

    return (np.array(max_loc, dtype=int) / scale + np.array([x, y], dtype=int)).astype(int).tolist()

def trackbar_exists(name, window):
    try:
        cv.getTrackbarPos(name, window)
        return True
    except cv.error:
        return False

def draw_buttons(template_trackers, selected_template, grid_cols, button_size):
    if not template_trackers:
        cv.imshow("Controls", np.zeros((50, 150, 3), dtype=np.uint8))
        return

    grid_rows = (len(template_trackers) + grid_cols - 1) // grid_cols
    panel_height = grid_rows * button_size
    panel_width = grid_cols * button_size
    button_img = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

    for i, tracker in enumerate(template_trackers):
        row = i // grid_cols
        col = i % grid_cols
        x_offset = col * button_size
        y_offset = row * button_size

        # Resize template to exactly match button size (no centering, just direct fit)
        template_resized = cv.resize(tracker.template, (button_size, button_size))

        # Directly place the resized template inside its button
        button_img[y_offset:y_offset + button_size, x_offset:x_offset + button_size] = template_resized

        # Draw button border
        if i == selected_template: color = (255, 0, 0)
        elif template_trackers[i].start_hold or template_trackers[i].end_hold: color = (0, 255, 0)
        elif template_trackers[i].hold_used: color = (0, 255, 255)
        else: color = (0, 0, 0)
        cv.rectangle(button_img, (x_offset, y_offset), (x_offset + button_size, y_offset + button_size), color, 2)

        # Add button index
        cv.putText(button_img, str(i + 1), (x_offset + 5, y_offset + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv.imshow("Controls", button_img)


"""
Main function to run the template tracker
@param vidfile: str - path to video file
@param scale_factor: float - factor to resize the video by
@param output_annotated: bool - whether to output annotated video
@param annotated_stops: bool - whether to output includes all video pauses or not. True if include pauses, False if not
"""
def main(vidfile: str, scale_factor: float, output_annotated=False, annotated_stops=False):
    frames, original_size, fps = load_all_frames(vidfile, scale_factor)
    current_frame = 0
    playing = False
    tracking = False

    ############ Adjustable parameters ############
    playback_speed = 0.2 # 1.0 is normal speed
    search_radius = 30
    confidence_threshold = 0.75
    jump_frames = 10
    sample_interval = 50
    json_out_directory = "JsonOut"
    annotated_videos_directory = "AnnotatedVideos"
    ###############################################

    template_trackers = []
    selected_template = None
    actions = ["Change Template", "Hold Used", "Start Hold", "End Hold", "New Template"]
    current_action = "New Template"
    grid_cols = 5
    button_size = 80

    original_width, original_height = original_size
    aspect_ratio = original_width / original_height
    start_frame = None
    end_frame = None
    tracking_data = {}

    #region Section1: Mouse and Trackbar Callbacks
    def handleMouse(event, x, y, flags, param):
        nonlocal playing, current_action, selected_template
        if not playing and current_action != "Change Template":
            if event == cv.EVENT_LBUTTONDOWN:
                param['bbox_topleft'] = (x, y)
            if event == cv.EVENT_LBUTTONUP:
                if (x,y) == param['bbox_topleft']: #Ignore if no movement in bounding box
                    return
                if x < param['bbox_topleft'][0] or y < param['bbox_topleft'][1]:
                    return
                bbox_bottomright = (x, y)
                template = extract_template(frames[current_frame], param['bbox_topleft'], bbox_bottomright)
                template_trackers.append(TemplateTracker(template, param['bbox_topleft']))
        else:
            if not playing and current_action == "Change Template" and selected_template is not None:
                if event == cv.EVENT_LBUTTONDOWN:
                    param['bbox_topleft'] = (x, y)
                if event == cv.EVENT_LBUTTONUP:
                    if (x, y) == param['bbox_topleft']:  # Ignore if no movement in bounding box
                        return
                    if x < param['bbox_topleft'][0] or y < param['bbox_topleft'][1]:
                        return
                    bbox_bottomright = (x, y)
                    template = extract_template(frames[current_frame], param['bbox_topleft'], bbox_bottomright)
                    frame, hold_used = template_trackers[selected_template].get_hold_used()
                    start_hold = template_trackers[selected_template].get_start_hold()
                    end_hold = template_trackers[selected_template].get_end_hold()
                    template_trackers[selected_template] = TemplateTracker(template, param['bbox_topleft'])
                    if hold_used:
                        template_trackers[selected_template].set_hold_used(frame)
                    if start_hold:
                        template_trackers[selected_template].set_start_hold()
                    if end_hold:
                        template_trackers[selected_template].set_end_hold()

    # Trackbar callback
    def on_trackbar_width(val):
        nonlocal original_width, original_height
        window_width = max(100, cv.getTrackbarPos("Width", "Controls"))
        window_height = int(window_width / aspect_ratio)
        if trackbar_exists("Height", "Controls"):
            cv.setTrackbarPos("Height", "Controls", window_height)
        cv.resizeWindow("GUI", window_width, window_height)

    def on_trackbar_height(val):
        nonlocal original_width, original_height
        window_height = max(100, cv.getTrackbarPos("Height", "Controls"))
        window_width = int(window_height * aspect_ratio)
        if trackbar_exists("Width", "Controls"):
            cv.setTrackbarPos("Width", "Controls", window_width)
        cv.resizeWindow("GUI", window_width, window_height)

    def on_trackbar_speed(val):
        nonlocal playback_speed
        playback_speed = max(0.05, val / 5.0)

    def on_trackbar_jump(val):
        nonlocal jump_frames
        jump_frames = val

    def on_trackbar_action(val):
        nonlocal current_action
        current_action = actions[val]

    def on_mouse_controls(event, x, y, flags, param):
        nonlocal selected_template, current_action, current_frame, playing, grid_cols, button_size
        if event == cv.EVENT_LBUTTONDOWN:
            col = x // button_size
            row = y // button_size
            index = row * grid_cols + col
            if 0 <= index < len(template_trackers):
                selected_template = index
                if current_action == "New Template":
                    draw_buttons(template_trackers, selected_template, grid_cols, button_size)
                elif current_action == "Hold Used" and not playing:
                    template_trackers[selected_template].set_hold_used(current_frame)
                elif current_action == "Start Hold" and not playing:
                    template_trackers[selected_template].set_start_hold()
                elif current_action == "End Hold" and not playing:
                    template_trackers[selected_template].set_end_hold()
    #endregion

    #region Section2: Controls and gui setup
    cv.namedWindow("GUI", cv.WINDOW_NORMAL) # widnow for the video
    cv.namedWindow("Controls", cv.WINDOW_NORMAL) # all controls in this window to prevent resizing issues

    cv.setMouseCallback("GUI", handleMouse, {'bbox_topleft': None})
    cv.setMouseCallback("Controls", on_mouse_controls)

    # Create trackbars that control the size of the GUI window. Keeps aspect ratio and assumes vertical filming
    cv.createTrackbar("Width", "Controls", original_width, 1080, on_trackbar_width)
    cv.createTrackbar("Height", "Controls", original_height, 1920, on_trackbar_height)
    cv.createTrackbar("Speed", "Controls", int(playback_speed * 10), 10, on_trackbar_speed)
    cv.createTrackbar("Confidence", "Controls", int(confidence_threshold * 100), 100, lambda v: None)
    cv.createTrackbar("Radius", "Controls", search_radius, 500, lambda v: None)
    cv.createTrackbar("Jump", "Controls", jump_frames, 100, on_trackbar_jump)
    cv.createTrackbar("Actions", "Controls", 4, len(actions) - 1, on_trackbar_action)
    #endregion

    last_time = time.time()

    # Create directory for json output.
    os.makedirs(json_out_directory, exist_ok=True)
    base_file_name = os.path.splitext(os.path.basename(vidfile))[0]
    file_path_json = os.path.join(json_out_directory, f"{base_file_name}.json")

    # Write annotated frames to video
    if output_annotated:
        os.makedirs(annotated_videos_directory, exist_ok=True)
        output_video_path = os.path.join(annotated_videos_directory, f"{base_file_name}_annotated.mp4")
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        fps = 30
        out = cv.VideoWriter(output_video_path, fourcc, fps, (original_width, original_height))


    while True:
        frame_to_display = frames[current_frame].copy()
        overlay = frame_to_display.copy()

        confidence_threshold = cv.getTrackbarPos("Confidence", "Controls") / 100.0
        search_radius = cv.getTrackbarPos("Radius", "Controls")
        draw_buttons(template_trackers, selected_template, grid_cols, button_size)
        frame_data = {}

        for i, tracker in enumerate(template_trackers):
            search_region, loc = tracker.update_location(frame_to_display, confidence_threshold, search_radius)

            if loc is not None:
                frame_data[f"Hold_{i+1}"] = {"x": loc[0], "y": loc[1], "visible": True,
                                           **({"frame_used": tracker.hold_used_frame} if tracker.hold_used else {}),
                                           **({"start_hold": tracker.start_hold} if tracker.start_hold else {}),
                                           **({"end_hold": tracker.end_hold} if tracker.end_hold else {})}
                if i == selected_template: color = (255, 0, 0)
                elif tracker.start_hold or tracker.end_hold: color = (0, 255, 0)
                elif tracker.hold_used: color = (0, 255, 255)
                else: color = (0, 0, 255)
                cv.rectangle(overlay, loc,
                             (loc[0] + tracker.template.shape[1], loc[1] + tracker.template.shape[0]), color, 1)
                cv.putText(overlay, str(i + 1), (loc[0], loc[1] - 1), cv.FONT_HERSHEY_SIMPLEX, 0.4, color,
                           1)
                # Uncomment to see search areas: Draw search region in green
                # x, y, w, h = search_region
                # cv.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                frame_data[f"Hold_{i + 1}"] = {"visible": False,
                                               **({"frame_used": tracker.hold_used_frame} if tracker.hold_used else {}),
                                               **({"start_hold": tracker.start_hold} if tracker.start_hold else {}),
                                               **({"end_hold": tracker.end_hold} if tracker.end_hold else {})}

        cv.addWeighted(overlay, 0.5, frame_to_display, 0.5, 0, frame_to_display)
        cv.putText(frame_to_display, current_action, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow("GUI", frame_to_display)

        # Write annotated frames to video
        if output_annotated and annotated_stops:
            out.write(frame_to_display)

        if playing:
            # Write annotated frames to video
            if output_annotated and not annotated_stops:
                out.write(frame_to_display)

            if time.time() - last_time >= (1/30) / playback_speed: #Control frame speed
                current_frame += 1
                last_time = time.time()
            if current_frame == len(frames):
                break
                # current_frame = 0

        if current_frame % sample_interval == 0 or current_frame == len(frames)-1:
            tracking_data[f"frame_{current_frame}"] = frame_data

        key = cv.waitKey(1)
        # Quit on Q
        if key == ord("q"):
            break

        # Play/Pause on Space
        if key == ord(" "):
            playing = not playing

        # Mark start of climb
        if key == ord("s"):
            start_frame = current_frame

        # Mark end of climb
        if key == ord("e"):
            end_frame = current_frame

        # jump forawrd on right arrow
        if key == ord("d"):
            if not playing and current_frame + jump_frames < len(frames):
                current_frame += jump_frames

        # jump backward on left arrow
        if key == ord("a"):
            if not playing and current_frame - jump_frames >= 0:
                current_frame -= jump_frames

    if output_annotated:
        out.release()
    cv.destroyAllWindows()

    # Get rest of input
    route_color = input("Enter the color of the route: ")
    route_rating = input("Enter the grade of the route: ")
    climb_valid = input("Was the climb valid? Enter 'complete', 'fell', 'touch ground', or 'touch wrong hold': ")
    climber_id = input("Enter the climber's ID: ")
    climb_id = input("Enter the climb ID: ")

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    json_output = {
        "video_file": base_file_name,
        "timestamp": current_time,
        "fps": fps,
        "climber_id": climber_id,
        "climb_id": climb_id,
        "total_frames": len(frames),
        "sample_interval": sample_interval,
        "number_of_holds": len(template_trackers),
        "color_of_route": route_color,
        "route_rating": route_rating,
        "climb_valid": climb_valid,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "tracking_data": tracking_data
    }


    with open(file_path_json, "w") as json_file:
        json.dump(json_output, json_file, indent=2)

    print(f"Tracking data saved to {file_path_json}")

    if output_annotated:
        print(f"Annotated video saved to {output_video_path}")


# Parameters:
# vidfile: str - path to video file
# scale_factor: float - factor to resize the video by
# output_annotated: bool - whether to output annotated video
# annotated_stops: bool - whether to output includes all video pauses or not
if __name__ == "__main__":
    main("Data/Data83.MOV", 0.5, False, False)