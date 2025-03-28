# BouleringDB Annotation Tool

## Overview
This Annotation Tool is designed to track holds of a video for bouldering. It uses OpenCV to track the bounding boxes of the holds.
The tool allows you to draw bounding boxes around the holds and then track them throughout the video. All hold information 
and other route related information will be saved to a json file.

## Installation
Make sure to use Python 3.10 or lower. OpenCV is not compatible with new Python versions.

Install the requirments.txt file using the following command:
```bash
pip install -r requirements.txt
```

## Usage
Run the following command to start the Annotation Tool:
```bash
python main.py <video_path> <scale_factor>
```

This may take a minute to load. Once it has loaded there will be two screens, the control panel and the video.
The video shows a crosshair. You can use this to draw bounding boxes. Once a bounding box is drawn, it will show up 
in the control pannel. Additionally the current action you are using is shown in the top left of the video. You can adjust the 
action in the control pannel. The control pannel consists of tools to help you annotate the video. To apply an action to 
a template you first select the action then click on the template in the control pannel. If you are on the New Template
action you can freely click on templates in the control pannel. They will be highlighted in blue when selected.
After annotating the video you must enter two text inputs: The color of the route and the grade of the route. There
is an optional third for tracking climber id.

### Actions


#### Drawing a Template (New Template)
To draw a template, click and drag the mouse to create a rectangle. This will create a template that will be used to track the hold.

#### Setting the End Hold (End Hold)
To set the end hold, make sure you are on the end hold action then click on the template in the control panel.
The hold should turn green after you select a new template. You can undo this action by clicking on the template again
while on the end hold action.

#### Setting the Start Hold (Start Hold)
To set the start hold, make sure you are on the start hold action then click on the template in the control panel.
The hold should turn green after you select a new template. You can undo this action by clicking on the template again
while on the start hold action.

#### Mark a Hold as Used (Hold Used)
To mark a hold as used, make sure you are on the hold used action then click on the template in the control panel.
The hold should turn yellow after you select another hold. You can undo this action by clicking on the template again while 
on the hold used action.

#### Changing a Template (Change Template)
To change a template, make sure you are on the change template action then click on the template in the control panel.
This will allow you to redraw the template. After drawing the new bounding box you should see the new template reflected
in the control panel. If you had already marked that hold as a start hold, end hold, or used hold, the new template will
retain that information.


### Control Panel
The control panel consists of the following sliders:

#### Width and Height
These sliders allow you to adjust the width and height of the video. I will keep the aspect ratio
as long as the video fits on your screen.

#### Speed
This slider allows you to adjust the speed of the video. You can't make the video faster but you can 
slow it down.

#### Confidence
This slider allows you to adjust the confidence of the bounding box. This is a threshold for similarity when
doing the template matching. The higher the confidence the more similar the template must be to the bounding box.

#### Radius
This slider allows you to adjust the search radius for the template matching. This is the distance the template
can move from the original bounding box (bounding boxes are calculated from the top-left).

#### Jump
This slider allows you to adjust the number of frames the video will jump when you press the next or previous button.

#### Actions
This slider allows you to select the action you want to perform on the templates. The actions are discussed above.


### Hot Keys
These should only be used when the video is paused.
- **Space**: Play/Pause the video.
- **q**: Quit the video annoatater.
- **s**: Mark the current frame as the start of the climb.
- **e**: Mark the current frame as the end of the climb.
- **d**: Jump forward in the video (like a fastforward).
- **a**: Jump back in the video (like a rewind).