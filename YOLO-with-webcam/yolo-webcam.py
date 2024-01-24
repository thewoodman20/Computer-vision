from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720) #resolution of camera/window size, scaling up gives you a bigger screen

model = YOLO("../YOLO/YOLO-Weights/yolov8n.pt")


while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #bounding box
            x1, y1, x2, y2 = box.xyxy[0] #box.xywh - xy, width height
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2),(255,0,255), 3)
            
            #confidence
            conf = math.ceil((box.conf[0]*100))/100
            cvzone.putTextRect(img, f'{conf}', (x1, y1-20))
            #class name
            
            
    cv2.imshow("Image", img)
    cv2.waitKey(1)