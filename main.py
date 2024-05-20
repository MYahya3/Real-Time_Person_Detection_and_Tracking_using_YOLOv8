import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

## Load YOLOv8-Nano pre-trained model ##
model = YOLO("yolov8n.pt")
## Intialize Video Capture ##
cap = cv2.VideoCapture("testing.mp4")
## Check to make sure reading video ##
assert cap.isOpened(), "Error reading video file"

# While Loop to get frame-by-by frame from video
while cap.isOpened():
    success, frame = cap.read()  # Read frames
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    # Initialize Annotator to draw boxes on detections
    annotator = Annotator(frame)
    # Make model predictions on each frame for specific class_id = 0: person
    tracks = model.track(frame, persist=True, show=False, classes = 0)

    # Annotator Init and region drawing
    if tracks[0].boxes.id is not None:
        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        track_ids = tracks[0].boxes.id.int().cpu().tolist()
        # Draw boxes on detections
        for box, track_id, cls in zip(boxes, track_ids, clss):
            annotator.box_label(box, label=f"{model.names[int(cls)]}", color=colors(int(track_id), True))

    """Display frame."""
    cv2.imshow("Frane", frame)
    # Break Window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
