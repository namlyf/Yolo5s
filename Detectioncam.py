import torch
import cv2
import numpy as np

# YOLOv5s
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    

    results = model(frame)
    detections = results.pandas().xyxy[0]

    for _, row in detections.iterrows():
        x1, y1, x2, y2, confidence, class_id = row[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']]
        label = results.names[int(class_id)]

        if confidence > 0.4:  
            color = (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    cv2.imshow("Object Detection", frame)


    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
