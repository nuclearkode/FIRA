import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import picamera
import picamera.array
import tensorflow as tf
from sklearn.linear_model import RANSACRegressor
from kalman_filter import KalmanFilter  # Assuming Kalman filter implementation in kalman_filter.py

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.OUT)  # Forward
GPIO.setup(24, GPIO.OUT)  # Backward

# Servo setup
SERVO_PIN = 18
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM frequency
servo.start(0)

# Camera setup
camera = picamera.PiCamera()
camera.resolution = (400, 240)
camera.framerate = 20
rawCapture = picamera.array.PiRGBArray(camera, size=(400, 240))

# Perspective transformation matrix
source_points = np.float32([[80, 160], [300, 160], [40, 210], [340, 210]])
destination_points = np.float32([[130, 0], [310, 0], [130, 240], [310, 240]])
matrix = cv2.getPerspectiveTransform(source_points, destination_points)

# Initialize Kalman Filter for sensor fusion
kalman_filter = KalmanFilter()

# Load deep learning models
lane_detection_model = tf.keras.models.load_model('lane_detection_model.h5')

# YOLO Object Detection
def load_yolo_model():
    return tf.saved_model.load('object_detection_model')

object_detection_model = load_yolo_model()

# PID Controller Parameters
Kp = 0.1
Ki = 0.01
Kd = 0.01
prev_error = 0
integral = 0

# Adaptive PID parameters
Kp_high_error = 0.2

def capture_frame():
    camera.capture(rawCapture, format="bgr")
    frame = rawCapture.array
    rawCapture.truncate(0)
    return frame

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = adaptive_histogram_equalization(gray)
    smoothed = cv2.GaussianBlur(equalized, (5, 5), 0)
    return smoothed

def adaptive_histogram_equalization(frame):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(frame)

def perspective_transform(frame):
    warped = cv2.warpPerspective(frame, matrix, (400, 240))
    return warped

def threshold_frame(frame):
    _, binary = cv2.threshold(frame, 160, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary, 50, 150)
    return edges

def hough_lane_detection(edges):
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=50)
    if lines is None:
        return None, None
    
    # Extract lane lines and calculate average slope and intercept
    left_lines, right_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue  # Avoid division by zero
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        if slope < 0:  # Left lane
            left_lines.append((slope, intercept))
        else:  # Right lane
            right_lines.append((slope, intercept))
    
    # Average left and right lane lines
    def average_line(lines):
        if len(lines) > 0:
            slope_avg = np.mean([line[0] for line in lines])
            intercept_avg = np.mean([line[1] for line in lines])
            return slope_avg, intercept_avg
        else:
            return None, None
    
    left_slope, left_intercept = average_line(left_lines)
    right_slope, right_intercept = average_line(right_lines)
    
    return (left_slope, left_intercept), (right_slope, right_intercept)

def yolo_object_detection(frame):
    # Perform YOLO object detection
    detections = object_detection_model(frame)
    return detections

def pid_control(error):
    global prev_error, integral
    integral += error
    derivative = error - prev_error
    prev_error = error
    
    if abs(error) > 50:
        control = Kp_high_error * error + Ki * integral + Kd * derivative
    else:
        control = Kp * error + Ki * integral + Kd * derivative
    
    return control

def control_car(control):
    # Convert control value to servo angle (assuming control range is -30 to 30 degrees)
    angle = np.clip(90 + control, 60, 120)
    duty_cycle = (angle / 18.0) + 2.5
    servo.ChangeDutyCycle(duty_cycle)

    if control < -10:
        GPIO.output(21, True)
        GPIO.output(24, False)
        direction = "Left"
    elif -10 <= control < -5:
        GPIO.output(21, True)
        GPIO.output(24, False)
        direction = "Slight Left"
    elif -5 <= control < 5:
        GPIO.output(21, True)
        GPIO.output(24, False)
        direction = "Forward"
    elif 5 <= control < 10:
        GPIO.output(21, True)
        GPIO.output(24, False)
        direction = "Slight Right"
    elif control >= 10:
        GPIO.output(21, True)
        GPIO.output(24, False)
        direction = "Right"
    return direction

def display_info(frame, fps, control, direction):
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Control: {control:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Direction: {direction}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Processed Frame", frame)

def main():
    try:
        while True:
            start_time = time.time()

            frame = capture_frame()
            
            # Perform deep learning-based lane detection
            lane_detected_frame = deep_learning_lane_detection(frame)
            
            # Perform YOLO object detection
            detections = yolo_object_detection(frame)
            
            # Perform perspective transform on lane detection frame
            warped_frame = perspective_transform(lane_detected_frame)
            
            # Threshold the transformed frame to find lane edges
            edges_frame = threshold_frame(warped_frame)

            # Find lane points using Hough Transform
            lane_lines = hough_lane_detection(edges_frame)
            
            # Update Kalman filter with lane center measurement
            if lane_lines is not None:
                # Calculate lane center using the detected lines
                left_line, right_line = lane_lines
                if left_line is not None and right_line is not None:
                    left_slope, left_intercept = left_line
                    right_slope, right_intercept = right_line
                    lane_center = (left_intercept + right_intercept) / 2
                else:
                    lane_center = edges_frame.shape[1] // 2  # Use default center
                kalman_filter.update(np.array([[lane_center]]))
            else:
                kalman_filter.update(np.array([[edges_frame.shape[1] // 2]]))

            # Predict lane center using Kalman filter
            lane_center = kalman_filter.predict()[0, 0]
            frame_center = edges_frame.shape[1] // 2
            diff = lane_center - frame_center

            # Compute control output using PID controller
            control = pid_control(diff)
            
            # Adjust car steering based on control output
            direction = control_car(control)

            # Calculate FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)

            # Display processed frame with information
            display_info(frame, fps, control, direction)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        servo.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
