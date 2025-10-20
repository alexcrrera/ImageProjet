import cv2
import numpy as np
import os

print("Current working directory:", os.getcwd())
print("File exists:", os.path.exists("assets/videos/test.mp4"))

video_dir = "assets/videos/test4.mp4"  # ✅ no leading slash

# Load the video
video = cv2.VideoCapture(video_dir)
if not video.isOpened():
    print("❌ Could not open video:", video_dir)
    exit()

# Background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=True)



while True:
    ret, frame = video.read()
    if not ret:
        
        break

    # Apply background subtraction
    fgMask = backSub.apply(frame)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=2)
    fgMask = cv2.dilate(fgMask, kernel, iterations=2)

    # Find contours of moving objects
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Choose the largest moving object (likely the drone)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 100:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Extract and zoom ROI
            roi = frame[y:y + h, x:x + w]
            if roi.size > 0:
                zoomed = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_CUBIC)
                cv2.imshow("ROI Zoom", zoomed)

    # Show other windows
    cv2.imshow("Frame", frame)
    cv2.imshow("Foreground Mask", fgMask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
 
        break

video.release()
cv2.destroyAllWindows()
