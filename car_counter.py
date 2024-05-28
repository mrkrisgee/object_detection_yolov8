from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

import torch

### For Video
cap = cv2.VideoCapture("Videos/cars.mp4")

model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("masks/mask_cars.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]
totalCount = []

while True:
    success, img = cap.read()
    # Resize image to match the size of our video
    resized_mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    imgRegion = cv2.bitwise_and(img, resized_mask)

    # Model results
    results = model(imgRegion, stream=True)

    # Init detections for sorting function
    detections = np.empty((0, 5))

    # Iterate through results
    for r in results:
        boxes = r.boxes
        # Iterate through boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            # Convert to int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Width and Height
            w, h = x2-x1, y2-y1           

            # Confidence
            conf = math.ceil((box.conf[0]*100))/100 # or conf = f"{box.conf[0]:.2f}"
            # Class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Define vehicle and confidence limits
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))        

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,0,255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2-x1, y2-y1
        # Draw rectangle
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        # Draw id
        cvzone.putTextRect(img, f"{int(id)}", (max(0, x1), max(35, y1)), scale=1, thickness=2, offset=10, font=cv2.FONT_HERSHEY_SIMPLEX)

        # Find center point
        cx, cy = x1+w//2,y1+h//2
        # Draw circle
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                # Draw green line when count goes up
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # Draw toal count text
    cvzone.putTextRect(img, f'Count: {len(totalCount)}', (50, 50), scale=1, thickness=2, offset=10, font=cv2.FONT_HERSHEY_SIMPLEX)    

    cv2.imshow("Image", img)
    cv2.waitKey(1)

