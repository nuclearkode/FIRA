import RPi.GPIO as GPIO
import serial
import time
import threading
import csv
import math

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor Pins (adjust based on your hardware)
MOTOR_LEFT_FORWARD = 17
MOTOR_LEFT_BACKWARD = 18
MOTOR_RIGHT_FORWARD = 22
MOTOR_RIGHT_BACKWARD = 23
PWM_FREQ = 100

GPIO.setup(MOTOR_LEFT_FORWARD, GPIO.OUT)
GPIO.setup(MOTOR_LEFT_BACKWARD, GPIO.OUT)
GPIO.setup(MOTOR_RIGHT_FORWARD, GPIO.OUT)
GPIO.setup(MOTOR_RIGHT_BACKWARD, GPIO.OUT)

left_pwm = GPIO.PWM(MOTOR_LEFT_FORWARD, PWM_FREQ)
right_pwm = GPIO.PWM(MOTOR_RIGHT_FORWARD, PWM_FREQ)
left_pwm.start(0)
right_pwm.start(0)

# Servo Pin (for steering)
SERVO_PIN = 12
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo_pwm = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz for servo
servo_pwm.start(7.5)  # Neutral position

# Ultrasonic Sensor Pins (front and side)
TRIG_PIN_FRONT = 5
ECHO_PIN_FRONT = 6
TRIG_PIN_SIDE = 7
ECHO_PIN_SIDE = 8
GPIO.setup(TRIG_PIN_FRONT, GPIO.OUT)
GPIO.setup(ECHO_PIN_FRONT, GPIO.IN)
GPIO.setup(TRIG_PIN_SIDE, GPIO.OUT)
GPIO.setup(ECHO_PIN_SIDE, GPIO.IN)

# Encoder Pins (for precise turns)
ENCODER_LEFT = 19
ENCODER_RIGHT = 20
GPIO.setup(ENCODER_LEFT, GPIO.IN)
GPIO.setup(ENCODER_RIGHT, GPIO.IN)

# Serial Setup for OpenMV communication
ser = serial.Serial('/dev/ttyS0', 115200, timeout=0.1)

# PID Parameters for Steering
Kp = 0.1
Ki = 0.01
Kd = 0.05
integral = 0
previous_error = 0
previous_time = time.time()

# Control Parameters
setpoint = 80  # Center of QQVGA (160 / 2)
base_speed = 50
curvature_threshold = 1000  # Adjust based on curvature units
error_speed_reduction_factor = 0.5
min_distance = 20  # cm (for obstacle avoidance)
low_confidence_threshold = 100  # Minimum pixel count for confidence
max_low_confidence = 10  # Frames before stopping
consecutive_low_confidence = 0

# Encoder variables for precise turns
left_encoder_count = 0
right_encoder_count = 0
counts_per_turn = 360  # Adjust based on encoder resolution and turn angle

def encoder_callback_left(channel):
    """Increment left encoder count on rising edge."""
    global left_encoder_count
    left_encoder_count += 1

def encoder_callback_right(channel):
    """Increment right encoder count on rising edge."""
    global right_encoder_count
    right_encoder_count += 1

# Set up encoder interrupts
GPIO.add_event_detect(ENCODER_LEFT, GPIO.RISING, callback=encoder_callback_left)
GPIO.add_event_detect(ENCODER_RIGHT, GPIO.RISING, callback=encoder_callback_right)

def set_steering_angle(angle):
    """Set servo angle (-90 to 90 degrees)."""
    duty_cycle = 7.5 + (angle / 18)  # Map -90 to 90 degrees to 2.5 to 12.5 duty cycle
    servo_pwm.ChangeDutyCycle(duty_cycle)

def set_motors_forward(speed):
    """Set both motors to move forward at given speed (0-100)."""
    GPIO.output(MOTOR_LEFT_BACKWARD, GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_BACKWARD, GPIO.LOW)
    left_pwm.ChangeDutyCycle(min(speed, 100))
    right_pwm.ChangeDutyCycle(min(speed, 100))

def stop_motors():
    """Stop all motors."""
    left_pwm.ChangeDutyCycle(0)
    right_pwm.ChangeDutyCycle(0)
    servo_pwm.ChangeDutyCycle(7.5)  # Reset steering to neutral

def execute_turn(direction, angle):
    """Execute a precise turn using encoder feedback."""
    global left_encoder_count, right_encoder_count
    left_encoder_count = 0
    right_encoder_count = 0
    
    if direction == "Left":
        set_steering_angle(-angle)
    elif direction == "Right":
        set_steering_angle(angle)
    else:
        set_steering_angle(0)
    
    set_motors_forward(base_speed * 0.5)  # Reduced speed for turning
    
    while left_encoder_count < counts_per_turn and right_encoder_count < counts_per_turn:
        time.sleep(0.01)
    
    stop_motors()

def get_distance(trig_pin, echo_pin):
    """Measure distance using ultrasonic sensor in cm."""
    GPIO.output(trig_pin, GPIO.LOW)
    time.sleep(0.00001)
    GPIO.output(trig_pin, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trig_pin, GPIO.LOW)
    
    start_time = time.time()
    while GPIO.input(echo_pin) == 0 and time.time() - start_time < 0.1:
        pass
    pulse_start = time.time()
    
    while GPIO.input(echo_pin) == 1 and time.time() - pulse_start < 0.1:
        pass
    pulse_end = time.time()
    
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # Speed of sound in cm/s
    return round(distance, 2)

def check_obstacles():
    """Check for obstacles using front and side ultrasonic sensors."""
    front_distance = get_distance(TRIG_PIN_FRONT, ECHO_PIN_FRONT)
    side_distance = get_distance(TRIG_PIN_SIDE, ECHO_PIN_SIDE)
    return front_distance < min_distance or side_distance < min_distance

def control_loop():
    """Main control loop with PID steering, speed modulation, and safety features."""
    global integral, previous_error, previous_time, consecutive_low_confidence
    
    with open('log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'lane_center', 'steering_angle', 'speed', 'confidence', 'sign', 'zebra'])
        
        while True:
            current_time = time.time()
            dt = current_time - previous_time
            previous_time = current_time
            
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                try:
                    lane_center, curvature, confidence, sign, zebra = line.split(',')
                    lane_center = float(lane_center)
                    curvature = float(curvature)
                    confidence = int(confidence)
                    zebra = int(zebra)
                    
                    error = lane_center - setpoint
                    
                    # Handle specific signs
                    if sign in ["Proceed Left", "Proceed Right"]:
                        direction = "Left" if sign == "Proceed Left" else "Right"
                        execute_turn(direction, 45)  # 45-degree turn, adjust as needed
                        continue
                    elif sign in ["Stop", "No Entry", "Dead End"]:
                        stop_motors()
                        print(f"Sign detected: {sign}. Stopping.")
                        time.sleep(2)  # Pause before resuming
                        continue
                    
                    # Handle zebra crossing
                    if zebra:
                        stop_motors()
                        print("Zebra crossing detected. Stopping.")
                        time.sleep(1)  # Brief stop
                        continue
                    
                    # PID calculation for steering
                    integral += error * dt
                    derivative = (error - previous_error) / dt if dt > 0 else 0
                    steering_angle = Kp * error + Ki * integral + Kd * derivative
                    steering_angle = max(-90, min(90, steering_angle))
                    previous_error = error
                    
                    # Speed modulation based on curvature and error
                    speed = base_speed
                    if curvature < curvature_threshold:
                        speed *= 0.7  # Reduce speed on sharp curves
                    if abs(error) > 20:
                        speed *= (1 - error_speed_reduction_factor * (abs(error) / 80))
                    
                    # Fail-safe for low confidence
                    if confidence < low_confidence_threshold:
                        consecutive_low_confidence += 1
                    else:
                        consecutive_low_confidence = 0
                    
                    if consecutive_low_confidence > max_low_confidence:
                        stop_motors()
                        print("Low confidence. Stopping.")
                        time.sleep(1)
                    elif check_obstacles():
                        stop_motors()
                        print("Obstacle detected. Stopping.")
                        time.sleep(0.5)
                    else:
                        set_steering_angle(steering_angle)
                        set_motors_forward(speed)
                    
                    # Log data
                    writer.writerow([current_time, lane_center, steering_angle, speed, confidence, sign, zebra])
                
                except ValueError:
                    stop_motors()
                    print("Invalid data received")
            else:
                stop_motors()
                print("No data received")
            
            time.sleep(0.033)  # ~30 Hz

# Start control loop in a separate thread
try:
    control_thread = threading.Thread(target=control_loop)
    control_thread.start()
    control_thread.join()
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    stop_motors()
    GPIO.cleanup()
    ser.close()
