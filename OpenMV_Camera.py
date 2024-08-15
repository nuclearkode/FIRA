import sensor, image, time, pyb, math
from pyb import Servo, Pin, Timer

# Camera setup
sensor.reset()
sensor.set_pixformat(sensor.RGB565)  # Set to RGB565 for color detection
sensor.set_framesize(sensor.QQVGA)   # 160x120 resolution
sensor.skip_frames(time=2000)
sensor.set_brightness(0)
sensor.set_contrast(0)
sensor.set_saturation(0)
sensor.set_auto_gain(False)          # must turn this off for color tracking
sensor.set_auto_whitebal(False)      # must turn this off for color tracking

# Servo setup on P7
servo = Servo(1)  # Servo connected to P7

# L298N Motor Driver Pins
# Motor A setup
motor_a_dir1 = Pin('P0', Pin.OUT_PP)  # IN1
motor_a_dir2 = Pin('P1', Pin.OUT_PP)  # IN2
motor_a_speed = Timer(4, freq=1000).channel(3, Timer.PWM, pin=Pin('P9'))  # ENA (PWM on P9)

# Motor B setup
motor_b_dir1 = Pin('P2', Pin.OUT_PP)  # IN3
motor_b_dir2 = Pin('P3', Pin.OUT_PP)  # IN4
motor_b_speed = Timer(4, freq=1000).channel(2, Timer.PWM, pin=Pin('P8'))  # ENB (PWM on P8)

# Function to set motor speeds and direction
def set_motor_speed(motor, speed):
    if motor == 'A':
        motor_a_speed.pulse_width_percent(speed)
    elif motor == 'B':
        motor_b_speed.pulse_width_percent(speed)

def set_motor_direction(forward=True):
    if forward:
        motor_a_dir1.high()
        motor_a_dir2.low()
        motor_b_dir1.high()
        motor_b_dir2.low()
    else:
        motor_a_dir1.low()
        motor_a_dir2.high()
        motor_b_dir1.low()
        motor_b_dir2.high()

# PID Controller Class
class PID:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0

    def compute(self, current_value):
        error = self.setpoint - current_value
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

# Initialize PID for steering control
pid = PID(kp=0.5, ki=0.1, kd=0.1, setpoint=80)  # Setpoint is the center of the image width

# Basic Kalman Filter Implementation
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, estimated_measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_measurement_variance = estimated_measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def update(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate

# Initialize Kalman filter
kalman_filter = KalmanFilter(process_variance=1, measurement_variance=1, estimated_measurement_variance=1)

# Basic Object Detection (simulating YOLO with simple blob detection)
def object_detection(img):
    blobs = img.find_blobs([(0, 100, -128, 127, -128, 127)], pixels_threshold=200, area_threshold=200, merge=True)
    if blobs:
        largest_blob = max(blobs, key=lambda b: b.pixels())
        return largest_blob
    return None

# Function to calculate steering angle from detected lane
def calculate_steering_angle(img):
    img.lens_corr(1.8)
    img.binary([(0, 100)])  # Adjust binary threshold based on lane color
    img.dilate(1)
    img.erode(1)
    edges = img.find_edges(image.EDGE_CANNY, threshold=(100, 200))

    # Hough Transform to find lines
    lines = img.find_lines(threshold=1000, theta_margin=25, rho_margin=25)

    max_length = 0
    best_line = None

    for line in lines:
        length = math.sqrt((line.x2() - line.x1())**2 + (line.y2() - line.y1())**2)
        if length > max_length:
            max_length = length
            best_line = line

    if best_line:
        # Midpoint of the detected line
        line_mid_x = (best_line.x1() + best_line.x2()) // 2
        return line_mid_x
    else:
        return 80  # Default to the center if no lane is detected

# Main loop
while(True):
    # pyb.LED(1).on()  # Turn on the internal LED to white

    img = sensor.snapshot()  # Take an image from the camera

    # Process top two-thirds for color detection
    top_img = img.copy().crop(roi=(0, 0, img.width(), 80))  # ROI: x, y, width, height
    detected_object = object_detection(top_img)
    if detected_object:
        print("Object detected at:", detected_object.cx(), detected_object.cy())
        # Draw a red pixelated box in the top right corner if an obstacle is detected
        img.draw_rectangle((img.width()-20, 0, 20, 20), color=(255, 0, 0), fill=True)

    # Process bottom one-third for lane detection
    bottom_img = img.copy().crop(roi=(0, 80, img.width(), 40))  # ROI: x, y, width, height
    bottom_img = bottom_img.to_grayscale()  # Convert to grayscale before edge detection
    lane_position = calculate_steering_angle(bottom_img)

    # Check if lane is detected
    lane_detected = lane_position != 80

    # Update Kalman filter with lane position
    lane_position_filtered = kalman_filter.update(lane_position)

    # Compute the steering angle using PID controller
    steering_correction = pid.compute(lane_position_filtered)
    steering_angle = int(steering_correction)  # Convert to integer value expected by the servo
    steering_angle = max(0, min(180, steering_angle))  # Ensure steering angle is within 0-180 degrees

    # Draw directional arrows based on steering angle
    if steering_angle > 90:  # Turn right
        img.draw_arrow(img.width()//2, img.height()-20, img.width()-10, img.height()-20, color=(0, 0, 255), thickness=2)
    elif steering_angle < 90:  # Turn left
        img.draw_arrow(img.width()//2, img.height()-20, 10, img.height()-20, color=(0, 0, 255), thickness=2)

    # Control servo motor based on steering angle
    servo.angle(steering_angle)

    # Set motor speed (you can adjust these values as needed)
    motor_a_speed_value = 50
    motor_b_speed_value = 50
    set_motor_speed('A', motor_a_speed_value)  # Motor A at 50% speed
    set_motor_speed('B', motor_b_speed_value)  # Motor B at 50% speed

    # Set motor direction (forward or backward)
    set_motor_direction(forward=True)

    # Display additional information on the output image
    img.draw_string(2, 2, "Speed: A={}%, B={}%" .format(motor_a_speed_value, motor_b_speed_value), color=(255, 255, 255), scale=1)
    img.draw_string(2, 12, "Servo Angle: {}°" .format(steering_angle), color=(255, 255, 255), scale=1)
    img.draw_string(2, 22, "Lane Detected: {}".format("Yes" if lane_detected else "No"), color=(255, 255, 255), scale=1)


    time.sleep_ms(100)  # Small delay to prevent overwhelming the system
