import cv2
from ultralytics import YOLO
from testing_file import TrackProcessor

## Load YOLOv8-Nano pre-trained model ##
model = YOLO("yolov8n.pt")

## Intialize Video Capture ##
cap = cv2.VideoCapture("testing.mp4")
## Check to make sure reading video ##
assert cap.isOpened(), "Error reading video file"

# Parameters
class_id = 0  ### Class_id 0 is for Person detection only
Tracker = TrackProcessor()
# Set the mouse callback function for the window

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', Tracker.mouse_callback)
# While Loop to get frame-by-by frame from video
while cap.isOpened():
    success, frame = cap.read()  # Read frames
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    # Make model predictions on each frame for specific class_id = 0: person
    tracks = model.track(frame, persist=True, show=False, classes = 0)

    frame = Tracker.extract_and_process_tracks(image=frame, tracks=tracks, classes_names= model.names)

    """Display frame."""
    cv2.imshow("Frame", frame)
    # Break Window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()