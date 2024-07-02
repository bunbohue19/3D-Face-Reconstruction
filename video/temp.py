import cv2

print(cv2.VideoCapture('skincare.mp4').get(cv2.CAP_PROP_FPS))