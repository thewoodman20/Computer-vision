from ultralytics import YOLO
import cv2

model = YOLO('../YOLO-Weights/yolov8l.pt') #n for nano weights, m for medium, l for large. Smaller = faster
results = model("test.png", show=True)
cv2.waitKey(0)

