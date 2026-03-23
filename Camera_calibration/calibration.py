import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve()
while ROOT.name != "Camera_calibration":
    ROOT = ROOT.parent

# video_path_left = ROOT /"videos"/"test1.mp4"
# video_path_center = ROOT /"videos"/"test2.mp4"

# #frame id that you want to read
# frame_id = 100
# save_frames = True

# def extract_frame(video_path, frame_id):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Cannot open video:{video_path}")

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     if frame_id >= total_frames:
#         print("Frame number exceeds total frames.")
#         cap.release()
#         return None
    
#     cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
#     ret, frame = cap.read()
#     cap.release()

#     if not ret:
#         print("Failed to read frame.")
#         return None
    
#     print(f"Extracted frame {frame_id},from {video_path}")
#     print(f"Resolution:",frame.shape[1], "x", frame.shape[0])

#     return frame

# # 读取帧
# frame_left = extract_frame(video_path_left, frame_id)
# frame_center = extract_frame(video_path_center, frame_id)

# if frame_left is not None and frame_center is not None:
#     cv2.imshow("Left Frame", frame_left)
#     cv2.imshow("Center Frame",frame_center)

#     if save_frames:
#         cv2.imwrite("left_frame.jpg",frame_left)
#         cv2.imwrite("center_frame.jpg",frame_center)



left_image_path = ROOT/"left_frame.jpg"
center_image_path = ROOT/"center_frame.jpg"
img_left = cv2.imread(left_image_path)
img_center = cv2.imread(center_image_path)

if img_left is None or img_center is None:
    print("Error loading images.")
    exit()

display_left = img_left.copy()
display_center = img_center.copy()

points_left = []
points_center = []

current_index = 0
click_mode = "left" 


def redraw_points():
    """Redraw both images from source buffers after undo."""
    global display_left, display_center
    display_left = img_left.copy()
    display_center = img_center.copy()

    for i, (x, y) in enumerate(points_left):
        cv2.circle(display_left, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(display_left, str(i),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

    for i, (x, y) in enumerate(points_center):
        cv2.circle(display_center, (x, y), 6, (255, 0, 0), -1)
        cv2.putText(display_center, str(i),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)


def undo_last_click():
    """Undo last user action and keep pair indexing/mode consistent."""
    global current_index, click_mode

    if not points_left and not points_center:
        print("Nothing to undo.")
        return

    # Case 1: LEFT was clicked, waiting for CENTER.
    if click_mode == "center" and len(points_left) > len(points_center):
        removed = points_left.pop()
        click_mode = "left"
        print(f"Undo LEFT pending point: ({removed[0]}, {removed[1]})")
        redraw_points()
        return

    # Case 2: Last completed pair should be removed.
    if points_left and points_center and len(points_left) == len(points_center):
        removed_left = points_left.pop()
        removed_center = points_center.pop()
        current_index = max(0, current_index - 1)
        click_mode = "left"
        print(
            "Undo pair "
            f"{current_index}: LEFT({removed_left[0]}, {removed_left[1]}) "
            f"CENTER({removed_center[0]}, {removed_center[1]})"
        )
        redraw_points()
        return

    # Fallback for unexpected mismatch.
    if len(points_left) > len(points_center):
        removed = points_left.pop()
        click_mode = "left"
        print(f"Undo unmatched LEFT point: ({removed[0]}, {removed[1]})")
    elif len(points_center) > len(points_left):
        removed = points_center.pop()
        current_index = max(0, current_index - 1)
        click_mode = "center"
        print(f"Undo unmatched CENTER point: ({removed[0]}, {removed[1]})")
    redraw_points()




def click_left(event, x, y, flags, param):
    global current_index, click_mode

    if event == cv2.EVENT_LBUTTONDOWN and click_mode == "left":
        points_left.append([x, y])
        cv2.circle(display_left, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(display_left, str(current_index),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        print(f"[LEFT]  Point {current_index}: ({x}, {y})")

        click_mode = "center"


def click_center(event, x, y, flags, param):
    global current_index, click_mode

    if event == cv2.EVENT_LBUTTONDOWN and click_mode == "center":
        points_center.append([x, y])
        cv2.circle(display_center, (x, y), 6, (255, 0, 0), -1)
        cv2.putText(display_center, str(current_index),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)

        print(f"[CENTER] Point {current_index}: ({x}, {y})")

        current_index += 1
        click_mode = "left"
    
cv2.namedWindow("Left Image")
cv2.namedWindow("Center Image")

cv2.setMouseCallback("Left Image", click_left)
cv2.setMouseCallback("Center Image", click_center)

print("Instructions:")
print("1. Click LEFT image first.")
print("2. Then click CENTER image.")
print("3. Repeat for all correspondences.")
print("Press 'u' to undo, 's' to save, 'q' to quit.")

while True:
    cv2.imshow("Left Image", display_left)
    cv2.imshow("Center Image", display_center)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('u'):
        undo_last_click()

    if key == ord('s'):
        if len(points_left) >= 30 and len(points_left) == len(points_center):
            np.savetxt("points_left.txt",
                    np.array(points_left, dtype=np.float32),
                    fmt="%.2f")

            np.savetxt("points_center.txt",
                    np.array(points_center, dtype=np.float32),
                    fmt="%.2f")

            print("Points saved.")
        else:
            print("Need at least 30 valid point pairs.")

    if key == ord('q'):
        break
