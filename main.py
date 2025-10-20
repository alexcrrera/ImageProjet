import torchvision.models as models
import cv2
import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import numpy as np




model = models.mobilenet_v2(pretrained=True)
model.eval()

print(model.eval())

# -----------------------------
# 1. Load pretrained CNN
# -----------------------------
def loadModel(device='cpu'):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 3)  # 3 classes
    model.eval().to(device)
    return model

# -----------------------------
# 2. Preprocess ROI for CNN
# -----------------------------
def preprocessROI(roi):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    return transform(image).unsqueeze(0)

# -----------------------------
# 3. Classify a moving object
# -----------------------------
def classifyObject(model, roi, device='cpu'):
    tensor = preprocessROI(roi).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
    classes = ['Drone', 'Bird', 'Background']
    return classes[predicted.item()]

# -----------------------------
# 4. Detect motion using background subtraction
# -----------------------------
def detectMovingObjects(frame, backSub):
    fg_mask = backSub.apply(frame)
    # Remove noise
    fg_mask = cv2.medianBlur(fg_mask, 5)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) > 500:  # filter small noise
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))
    return boxes, fg_mask

# -----------------------------
# 5. Process a video frame-by-frame
# -----------------------------
def processVideo(videoPath, model, device='cpu'):
    cap = cv2.VideoCapture(videoPath)
    backSub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, fg_mask = detectMovingObjects(frame, backSub)
        for (x, y, w, h) in boxes:
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            label = classifyObject(model, roi, device)
            color = (0, 255, 0) if label == 'Drone' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Detections', frame)
        cv2.imshow('Foreground Mask', fg_mask)

        if cv2.waitKey(30) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# 6. Main entry point
# -----------------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device used: ",device)
    model = loadModel(device)
    videoPath = 'video.mp4'  # path to your input video
    processVideo(videoPath, model, device)

if __name__ == "__main__":
    main()
