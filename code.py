import cv2
import numpy as np
import os
import RPi.GPIO as GPIO
import time

TRIG = 23  
ECHO = 24  

def setup_ultrasonic():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)
    GPIO.output(TRIG, False)
    time.sleep(2)

def get_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    
    start_time = time.time()
    stop_time = time.time()
    
    while GPIO.input(ECHO) == 0:
        start_time = time.time()
    
    while GPIO.input(ECHO) == 1:
        stop_time = time.time()
    
    elapsed_time = stop_time - start_time
    distance = (elapsed_time * 34300) / 2 
    return round(distance, 2) 

whT = 224
confThreshold = 0.3
nmsThreshold = 0.2

def ann(object_name, position, distance):
    text = f"{object_name} {position} at {distance} centimeters"
    print("Announcement:", text)  # Debug print
    os.system(f'espeak-ng "{text}"')

classfile = '/home/angel/eye/model/coco.names'
with open(classfile, 'r') as f:
    classname = f.read().strip().split("\n")


modelConfiguration = '/home/angel/eye/model/yolov3-tiny.cfg'
modelWeights = '/home/angel/eye/model/yolov3-tiny.weights'
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def objectFind(outputs, frame_width, distance):
    detected_objects = set()
    for output in outputs:
        for d in output:
            scores = d[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:
                center_x = int(d[0] * frame_width)

                if center_x < frame_width // 3:
                    position = "left"
                elif center_x > 2 * frame_width // 3:
                    position = "right"
                else:
                    position = "center"
                
                object_name = classname[classId].upper()
                if object_name not in detected_objects:
                    detected_objects.add(object_name)
                    ann(object_name, position, distance)  # Announce with distance

# Initialize video capture
cap = cv2.VideoCapture(0)
frame_skip = 10
frame_count = 0
setup_ultrasonic()

try:
    while cap.isOpened():
        distance = get_distance()
        print(f"Measured Distance: {distance:.2f} cm")

        if distance < 100:  # Only detect objects within 1 meter
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame_height, frame_width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
            net.setInput(blob)
            layerNames = net.getLayerNames()
            outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
            outputs = net.forward(outputNames)
            objectFind(outputs, frame_width, distance)  
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("Stopping...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
