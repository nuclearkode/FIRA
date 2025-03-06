import sensor, image, time, pyb, math

# Initialize camera
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QQVGA)  # 160x120 for efficiency
sensor.skip_frames(time=2000)

# UART setup for Raspberry Pi communication
uart = pyb.UART(3, 115200)

# Load sign templates (replace with actual image files)
templates = {
    "Proceed Left": image.Image("proceed_left.bmp"),
    "Proceed Right": image.Image("proceed_right.bmp"),
    "Proceed Forward": image.Image("proceed_forward.bmp"),
    "Stop": image.Image("stop.bmp"),
    "No Entry": image.Image("no_entry.bmp"),
    "Dead End": image.Image("dead_end.bmp")
}

# HSV Thresholds for lane detection (yellow and white)
yellow_threshold = (20, 40, 100, 255, 100, 255)
white_threshold = (0, 255, 0, 50, 200, 255)

# Region of Interest (ROI) - bottom half of the image
roi = (0, 60, 160, 60)

def find_lane_lines(img):
    """Detect lane lines using sliding windows and polynomial fitting."""
    # Convert to HSV and apply color thresholds
    hsv_img = img.to_hsv()
    yellow_blobs = hsv_img.find_blobs([yellow_threshold], roi=roi, merge=True)
    white_blobs = hsv_img.find_blobs([white_threshold], roi=roi, merge=True)
    
    # Create a binary mask for lanes
    lane_mask = image.Image(img.width(), img.height(), sensor.BINARY)
    if yellow_blobs:
        lane_mask.draw_image(yellow_blobs[0].binary(), color=1)
    if white_blobs:
        lane_mask.draw_image(white_blobs[0].binary(), color=1)
    
    # Sliding window parameters
    n_windows = 9
    margin = 20
    minpix = 50
    
    # Find lane pixels using histogram
    histogram = lane_mask.histogram(roi=roi)
    midpoint = len(histogram) // 2
    leftx_base = histogram.index(max(histogram[:midpoint]))
    rightx_base = histogram.index(max(histogram[midpoint:])) + midpoint
    
    # Set up windows
    window_height = math.floor(roi[3] / n_windows)
    nonzero = lane_mask.find_nonzero()
    left_lane_inds = []
    right_lane_inds = []
    
    current_leftx = leftx_base
    current_rightx = rightx_base
    
    for window in range(n_windows):
        win_y_low = roi[1] + roi[3] - (window + 1) * window_height
        win_y_high = win_y_low + window_height
        win_leftx_low = current_leftx - margin
        win_leftx_high = current_leftx + margin
        win_rightx_low = current_rightx - margin
        win_rightx_high = current_rightx + margin
        
        good_left_inds = [(x, y) for x, y in nonzero if win_leftx_low <= x < win_leftx_high and win_y_low <= y < win_y_high]
        good_right_inds = [(x, y) for x, y in nonzero if win_rightx_low <= x < win_rightx_high and win_y_low <= y < win_y_high]
        
        if len(good_left_inds) > minpix:
            current_leftx = int(sum(x for x, _ in good_left_inds) / len(good_left_inds))
        if len(good_right_inds) > minpix:
            current_rightx = int(sum(x for x, _ in good_right_inds) / len(good_right_inds))
        
        left_lane_inds.extend(good_left_inds)
        right_lane_inds.extend(good_right_inds)
    
    # Calculate confidence based on detected pixels
    confidence = len(left_lane_inds) + len(right_lane_inds)
    
    # Fit second-order polynomials to lane lines
    if left_lane_inds and right_lane_inds:
        left_fit = pyb.polyfit([y for _, y in left_lane_inds], [x for x, _ in left_lane_inds], 2)
        right_fit = pyb.polyfit([y for _, y in right_lane_inds], [x for x, _ in right_lane_inds], 2)
        
        # Calculate curvature and lane center at the bottom of the image
        y_eval = img.height() - 1
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / abs(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / abs(2 * right_fit[0])
        curvature = (left_curverad + right_curverad) / 2
        
        lane_center = (left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2] + 
                       right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]) / 2
    else:
        lane_center = img.width() / 2  # Default to center if no lanes detected
        curvature = float('inf')  # Infinite curvature indicates no curve
        confidence = 0
    
    return lane_center, curvature, confidence

def detect_sign(img):
    """Recognize street signs via template matching."""
    for sign_name, template in templates.items():
        if img.find_template(template, 0.7):  # Match threshold
            return sign_name
    return "None"

def detect_zebra(img):
    """Detect zebra crossings by finding parallel horizontal lines."""
    lines = img.find_lines(threshold=1000, theta_margin=5, rho_margin=5)
    horizontal_lines = [l for l in lines if 85 < l.theta() < 95]
    if len(horizontal_lines) > 3:
        rhos = sorted([l.rho() for l in horizontal_lines])
        spacings = [rhos[i+1] - rhos[i] for i in range(len(rhos)-1)]
        if all(5 < s < 15 for s in spacings):  # Consistent spacing
            return 1
    return 0

# Main loop
while True:
    img = sensor.snapshot()
    lane_center, curvature, confidence = find_lane_lines(img)
    sign = detect_sign(img)
    zebra = detect_zebra(img)
    uart.write(f"{lane_center:.2f},{curvature:.2f},{confidence},{sign},{zebra}\n")
    time.sleep(0.05)  # ~20 FPS
