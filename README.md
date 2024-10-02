import cv2
import numpy as np
import time

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Log file to record movements
log_file = open("car_log.txt", "w")

# Dummy functions to represent car movement
def move_forward():
    log_file.write(f"{time.ctime()}: Car moving forward\n")
    print("Car moving forward")

def turn_left():
    log_file.write(f"{time.ctime()}: Car turning left\n")
    print("Car turning left")

def turn_right():
    log_file.write(f"{time.ctime()}: Car turning right\n")
    print("Car turning right")

def stop_car():
    log_file.write(f"{time.ctime()}: Car stopped\n")
    print("Car stopped")

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = frame.shape[:2]
    roi_vertices = [
        (0, height),
        (width // 2, height // 2),
        (width, height)
    ]

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([roi_vertices], np.int32), 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Adjusted Hough parameters for better lane detection
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)

    direction = None
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            if x1 < width // 2 and x2 < width // 2:
                direction = "left"
            elif x1 > width // 2 and x2 > width // 2:
                direction = "right"
            else:
                direction = "forward"
    else:
        print("No lanes detected!")
   
    return frame, direction

def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    obstacle_detected = False
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if label == "car" or label == "person":
                obstacle_detected = True

    return frame, obstacle_detected

# Use the camera URL instead of 0
camera_url = "http://192.168.1.2:8080/video"  # Replace with your camera URL
cap = cv2.VideoCapture(camera_url)

# Timing variables
fps = 0
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame. Exiting...")
        break

    lane_frame, direction = detect_lanes(frame.copy())
    object_frame, obstacle_detected = detect_objects(frame.copy())

    combined_frame = cv2.addWeighted(lane_frame, 0.6, object_frame, 0.4, 0)

    # Control logic with enhanced logging
    if obstacle_detected:
        stop_car()
        print("Obstacle detected! Stopping the car.")
    else:
        if direction == "left":
            turn_left()
            print("Turning left.")
        elif direction == "right":
            turn_right()
            print("Turning right.")
        else:
            move_forward()
            print("Moving forward.")

    # Calculate and display FPS
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(combined_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the combined frame with lane and object detection
    cv2.imshow("Self-Driving Car", combined_frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == 27:
        print("Exiting the program.")
        break

# Close the log file and release resources
log_file.close()
cap.release()
cv2.destroyAllWindows()
