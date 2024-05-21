from ultralytics.utils.plotting import Annotator
import cv2
import time

class TrackProcessor:
    def __init__(self, line_thickness=2, exclude_red=True):
        self.line_thickness = line_thickness
        self.exclude_red = exclude_red
        self.colors = self.generate_bgr_colors()
        self.start_time = None
        self.selected_track_id = None
        self.boxes = []
        self.track_ids = []
    @staticmethod
    def generate_bgr_colors():
        """Generates a list of BGR colors, excluding shades of red if specified."""
        colors = [
            (0, 0, 0),  # Black
            (128, 0, 0),  # Navy
            (255, 0, 0),  # Blue
            (0, 255, 0),  # Green
            (128, 128, 0),  # Teal
            (255, 255, 0),  # Cyan/Aqua
            (0, 100, 0),  # Dark Green
            (128, 128, 0),  # Turquoise
            (205, 0, 0),  # Medium Blue
            (209, 206, 0),  # Dark Turquoise
            (170, 178, 32),  # Light Sea Green
            (127, 255, 0),  # Spring Green
            (87, 139, 46),  # Sea Green
            (255, 144, 30),  # Dodger Blue
            (255, 191, 0)  # Deep Sky Blue
            ]
        return colors

    def mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (x1, y1, x2, y2) in enumerate(self.boxes):
                print(id)
                if int(x1) <= x <= int(x2) and int(y1) <= y <= int(y2):
                    self.selected_track_id = self.track_ids[i]
                    self.start_time = time.time()
                    print(f"Selected track ID: {self.selected_track_id}")
                    break
    def extract_and_process_tracks(self, image, tracks, classes_names=None):
        """Extracts and processes tracks for object counting in a video stream."""

        annotator = Annotator(image, line_width=self.line_thickness)

        if tracks[0].boxes.id is not None:
            self.boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            self.track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # Extract tracks
            for box, track_id, cls in zip(self.boxes, self.track_ids, clss):
                # print(f"BOX: {box}, Track: {track_id}")
                if self.selected_track_id == track_id:
                    print(self.selected_track_id, track_id)
                    color = (0, 0, 255)  # Red for selected box
                    annotator.box_label(box, label=f"{classes_names[cls]}", color=color)
                    self.is_selected = True
                else:
                    try:
                        color = self.colors[int(track_id)]
                        annotator.box_label(box, label=f"{classes_names[cls]}", color=color)
                    except:
                        color = self.colors[int(track_id) % len(self.colors)]
                        annotator.box_label(box, label=f"{classes_names[cls]}", color=color)

                # Display the timer on the top left if a box is selected
                if self.start_time is not None and self.selected_track_id == track_id:
                    elapsed_time = int(time.time() - self.start_time)
                    # Put the current time on top
                    cv2.putText(image, f"Time-Tracker: {elapsed_time}", (20, 20),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (190, 215, 255), 1, cv2.LINE_AA)

        return image