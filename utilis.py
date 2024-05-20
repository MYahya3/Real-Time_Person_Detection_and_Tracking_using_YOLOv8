from ultralytics.utils.plotting import Annotator

def get_colors():
    colors = colors_without_red_bgr = [
        (0, 0, 0),        # Black
        (128, 0, 0),      # Navy
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (128, 128, 0),    # Teal
        (255, 255, 0),    # Cyan/Aqua
        (0, 100, 0),      # Dark Green
        (128, 128, 0),    # Turquoise
        (205, 0, 0),      # Medium Blue
        (209, 206, 0),    # Dark Turquoise
        (170, 178, 32),   # Light Sea Green
        (127, 255, 0),    # Spring Green
        (87, 139, 46),    # Sea Green
        (255, 144, 30),   # Dodger Blue
        (255, 191, 0)     # Deep Sky Blue
    ]
    return colors

def Draw_Bounding_Boxes(image, tracks, line_thickness= 2,  classes_names= None):
    """Extracts and processes tracks for object counting in a video stream."""

    # To get list of colors
    Colors_list = get_colors()
    # Annotator Init and region drawing
    annotator = Annotator(image, line_width=line_thickness)

    # Extract bboxes, track_ids and classes
    if tracks[0].boxes.id is not None:
        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        track_ids = tracks[0].boxes.id.int().cpu().tolist()

        # Extract tracks
        for box, track_id, cls in zip(boxes, track_ids, clss):
            # Draw bounding box with Exception having Random Colors
            try:
                color = Colors_list[int(track_id)]
                annotator.box_label(box, label=f"{classes_names[cls]}", color=color)
            except:
                color = Colors_list[int(track_id) % len(Colors_list)]
                annotator.box_label(box, label=f"{classes_names[cls]}", color=color)

    return image