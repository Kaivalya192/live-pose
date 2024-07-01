import cv2
import numpy as np
import pyrealsense2 as rs
import time

def create_mask():
    points = []
    mask_path = './mask.png'

    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image_display, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow("Image", image_display)

    def generate_mask(image, points):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 255)
        return mask

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        # Wait for 1 second to allow the camera to warm up
        time.sleep(1)
        # Wait for a coherent pair of frames: depth and color    
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            raise Exception("Could not capture color frame")

        # Convert image to numpy array
        image = np.asanyarray(color_frame.get_data())
        image_display = image.copy()

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", select_points)

        print("Click on the image to select points. Press Enter when done.")

        while True:
            cv2.imshow("Image", image_display)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                break

        mask = generate_mask(image, points)

        # Save the mask image
        cv2.imwrite(mask_path, mask)
        cv2.destroyAllWindows()

        return mask_path

    finally:
        # Stop streaming
        pipeline.stop()

if __name__ == "__main__":
    mask_file_path = create_mask()
    print(f"Mask saved at: {mask_file_path}")
