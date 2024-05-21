import cv2
from ultralytics import YOLO
from utilis import TrackProcessor

## Load YOLOv8-Nano pre-trained model ##
model = YOLO("yolov8n.pt")

## Intialize Video Capture ##
try:
    rtsp_url = 'rtsp://192.168.100.4:8080/h264_ulaw.sdp'
    # cap = cv2.VideoCapture(rtsp_url)
    # For Test Sample Vidoe
    cap = cv2.VideoCapture("input_video/sample.mp4")
except:
    print("Retry to initiate your IP webcam")
    raise ConnectionError

## Check to make sure reading video ##
assert cap.isOpened(), "Error reading video file"

# Initialize Tracker Class
Tracker = TrackProcessor()

# Parameters
class_id = 0  ### Class_id 0 is for Person detection only
# Set the mouse callback function for the window
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', Tracker.mouse_callback, {'frame_count': 5})
frame_count = 0

# While Loop to get frame-by-by frame from video
while cap.isOpened():
    success, frame = cap.read()  # Read frames
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    # Increment frame count
    frame_count += 1

    # Make model predictions on each frame for specific class_id = 0: person
    tracks = model.track(frame, persist=True, show=False, classes = 0)

    frame = Tracker.process_draw_Boundboxes(image=frame, tracks=tracks, frame_count=frame_count, classes_names= model.names)

    # Maintain the selection for a few frames even if detection is skipped
    Tracker.maintain_MisDetection(frame_count)

    """Display frame."""
    cv2.imshow("Frame", frame)
    # Break Window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()