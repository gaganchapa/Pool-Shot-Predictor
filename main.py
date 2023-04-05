import contextlib
import math
import cv2
import numpy as np


def locate_stick(image):
    # Crop image and define color range
    cropped_image = image[90:, :]
    lower_color = np.array([77, 44, 159])
    upper_color = np.array([100, 89, 213])
    kernel = np.ones((5, 5), np.uint8)

    # Apply filters to image
    filtered_image = apply_color_filter(cropped_image, lower_color, upper_color)
    processed_image = process_image(filtered_image)
    dilated_image = cv2.dilate(processed_image, kernel, iterations=1)
    thresholded_image = cv2.erode(dilated_image, kernel, iterations=1)

    # Find contours and stick object
    contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = -1
    stick_object = []

    for i in contours:
        area = cv2.contourArea(i)

        if 20 < area < 150:
            cv2.drawContours(filtered_image, i, -1, (172, 0, 196), 2)
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor > 2:
                # Find the stick
                if 18 > w > 5 and 18 > h > 5:
                    if area > max_area:
                        max_area = area
                        stick_object = [x, y + 90, w, h]

    if stick_object:
        cv2.rectangle(filtered_image, (stick_object[0], stick_object[1] - 90),
                      (stick_object[0] + stick_object[2], stick_object[1] - 90 + stick_object[3]), (255, 255, 255), 2)
        return stick_object


def calculate_trigonometric(degrees):
    angle_in_radians = math.radians(degrees)
    sine = math.sin(angle_in_radians)
    cosine = math.cos(angle_in_radians)

    if abs(cosine) < 1e-15:
        cosine = 0
    if abs(sine) < 1e-15:
        sine = 0

    return sine, cosine


# Function to find the position of the cue ball
def findCueBall(image):
    # Crop the image and apply color filtering and edge detection
    cropped = image[56:383, 38:757]
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([40, 9, 107]), np.array([72, 96, 210]))
    filtered = cv2.bitwise_and(cropped, cropped, mask=mask)
    processed = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY), (7, 7), 2), 50, 50)
    # Find the contours
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 400 < area < 500:
            cv2.drawContours(filtered, cnt, -1, (172, 0, 196), 2)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            obj_cor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            # Find the ball
            if obj_cor >= 8:
                return [x + 38, y + 56, w, h]


def apply_color_filter(image, lower_bound, upper_bound):
    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask that filters out colors outside of the specified bounds
    color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    return cv2.bitwise_and(image, image, mask=color_mask)


def find_colored_balls(image):
    cropped_img = image[110:375, 38:770]
    kernel = np.ones((5, 5), np.uint8)

    lower_values = [[13, 133, 132], [74, 33, 71], [61, 36, 70], [82, 71, 72], [0, 98, 70], [61, 36, 70], [120, 53, 116],
                    [0, 0, 0]]
    upper_values = [[66, 255, 255], [123, 255, 255], [79, 232, 255], [125, 255, 255], [17, 255, 255], [79, 232, 255],
                    [179, 255, 255], [179, 255, 255]]
    filtered_imgs = []

    for x in range(8):
        lower = np.array(lower_values[x])
        upper = np.array(upper_values[x])
        filtered_imgs.append(apply_color_filter(cropped_img, lower, upper))

    for f_img in filtered_imgs:
        img_processed = process_image(f_img)
        ball_count = 0
        found_balls = []
        dial = cv2.dilate(img_processed, kernel, iterations=1)
        thres = cv2.erode(dial, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for i in contours:
            area = cv2.contourArea(i)

            if 50 < area < 520:
                cv2.drawContours(f_img, i, -1, (172, 0, 196), 1)
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                obj_cor = len(approx)
                x, y, w, h = cv2.boundingRect(approx)

                if (
                        7 < obj_cor < 14
                        and 15 < h < 38
                        and 15 < w < 38
                        and abs(w - h) < 7
                ):
                    ball_count += 1
                    found_balls.append([x, y, w, h])
                    cv2.circle(f_img, (x + w // 2, y + h // 2), (w // 2), (255, 255, 0), 2)
                    cv2.putText(f_img, "Ball", (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX,
                                0.8, (255, 255, 255), 1)

        if ball_count == 1:
            x, y, w, h, r = found_balls[0][0], found_balls[0][1], found_balls[0][2], found_balls[0][3], (
                    found_balls[0][2] // 2 + found_balls[0][3] // 2) // 2

            if r >= 15:
                r = 14

            return [x + 38, y + 110, w, h, r]


def process_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur with a kernel size of 7x7 and standard deviation of 2
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 2)

    return cv2.Canny(blurred_image, 50, 50)


def locate_holes():
    return [
        [15, 40, 65, 85],
        [372, 35, 412, 75],
        [725, 39, 795, 78],
        [25, 372, 74, 408],
        [377, 376, 420, 413],
        [725, 368, 795, 410],
    ]


def get_hit_point(stick, cue_ball, average_radius, hit_points):
    stick_points = []
    hit_point = []
    cue_ball_x = cue_ball[0] + cue_ball[2] // 2
    cue_ball_y = cue_ball[1] + cue_ball[3] // 2

    average_radius.append((stick[2] // 2 + stick[3] // 2) // 2)
    radius = sum(average_radius) // len(average_radius)

    o_x = stick[0] + stick[2] // 2
    o_y = stick[1] + stick[3] // 2
    for ang in range(360):
        sine, cosine = get_sine_cosine(ang)
        p_x = int(cosine * radius)
        p_y = int(sine * radius)
        stick_points.append([o_x + p_x, o_y + p_y])

    min_distance = math.inf
    for t_point in stick_points:
        distance = math.sqrt((cue_ball_x - t_point[0]) ** 2 + (cue_ball_y - t_point[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            hit_point = t_point

    hit_points.append(hit_point)
    sum_x, sum_y = 0, 0
    for point in hit_points:
        sum_x += point[0]
        sum_y += point[1]
    return [sum_x // len(hit_points), sum_y // len(hit_points)]


def get_sine_cosine(angle):
    angle = math.radians(angle)
    return math.sin(angle), math.cos(angle)


# Draw the result on the image with given paths and color
def drawResult(paths, color, prediction, final, accuracy=0):
    for i, path in enumerate(paths):
        # Draw dotted line between consecutive points in path
        if i > 0:
            draw_dotted_line(imgCropped, (paths[i - 1][0], paths[i - 1][1]), (path[0], path[1]), color)
            # Draw filled circle at each point in path
            cv2.circle(imgCropped, (path[0], path[1]), 10, color, cv2.FILLED)

    # Draw filled rectangle with text indicating prediction
    prediction_text = "Prediction: In" if prediction else "Prediction: Out"
    cv2.putText(imgCropped, prediction_text, (315, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1000, 100, 800), 2)


# Define a function that computes the equation of a line given two points
def line_equation(point_a, point_b):
    # Extract the coordinates of the two points
    x1, y1 = point_a[0], point_a[1]
    x2, y2 = point_b[0], point_b[1]
    # Compute the slope of the line
    try:
        slope = (y2 - y1) / (x2 - x1)
    except ZeroDivisionError:
        slope = (y2 - y1) / (x2 + 1 - x1)
    # Compute the y-intercept of the line
    y_intercept = y1 - (slope * x1)
    # Return the slope and y-intercept as a tuple
    return slope, y_intercept


# Define a function that draws a dotted line between two points on an image
# def draw_dotted_line(image, point_a, point_b, color):
#     cv2.line(image, point_a, point_b, color, thickness=2)


def draw_dotted_line(image, point_a, point_b, color):
    # Compute the distance between the two points
    distance = ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5
    # Initialize an empty list to store the points along the line
    points = []
    # Iterate over the line with a step size of 15 pixels
    for i in range(0, int(distance), 15):
        # Compute the relative position of the current point along the line
        r = i / distance
        # Compute the x and y coordinates of the current point using linear interpolation
        x = int((point_a[0] * (1 - r) + point_b[0] * r) + 0.5)
        y = int((point_a[1] * (1 - r) + point_b[1] * r) + 0.5)
        # Add the current point to the list of points along the line
        points.append((x, y))
    # Draw circles at each point along the line to create a dotted effect
    for point in points:
        # cv2.circle(image, point, 3, color, -1)
        cv2.line(image,point_a,point_b,color,2)

# Function to detect collision between a cue ball and a colored ball
def detect_collision(cue_ball, colored_ball):
    # Create lists to store the points on the cue ball and colored ball
    cue_ball_points = []
    colored_ball_points = []

    # Calculate the radius and center of the cue ball
    cue_radius = (cue_ball[2] - cue_ball[0]) // 2
    cue_center_x = cue_ball[0] + (cue_ball[2] - cue_ball[0]) // 2
    cue_center_y = cue_ball[1] + (cue_ball[3] - cue_ball[1]) // 2

    # Calculate the points on the cue ball using polar coordinates
    for angle in range(0, 360):
        sine, cosine = get_sine_cosine(angle)
        point_x = int(cosine * cue_radius)
        point_y = int(sine * cue_radius)
        cue_ball_points.append([cue_center_x + point_x, cue_center_y + point_y])

    # Calculate the radius and center of the colored ball
    colored_radius = colored_ball[4]
    colored_center_x = colored_ball[0] + (colored_ball[2] - colored_ball[0]) // 2
    colored_center_y = colored_ball[1] + (colored_ball[3] - colored_ball[1]) // 2

    # Calculate the points on the colored ball using polar coordinates
    for angle in range(0, 360):
        sine, cosine = get_sine_cosine(angle)
        point_x = int(cosine * colored_radius)
        point_y = int(sine * colored_radius)
        colored_ball_points.append([colored_center_x + point_x, colored_center_y + point_y])

    # Find the points of intersection between the cue ball and colored ball
    intersection_points = []
    for point in cue_ball_points:
        if point in colored_ball_points:
            intersection_points.append(point)

    # If there are intersection points, calculate the average point and return it
    if len(intersection_points) > 0:
        avg_point = [0, 0]
        for point in intersection_points:
            avg_point[0] += point[0]
            avg_point[1] += point[1]
        avg_point[0] //= len(intersection_points)
        avg_point[1] //= len(intersection_points)
        cv2.circle(imgCropped, (avg_point[0], avg_point[1]), 8, (0, 200, 200), cv2.FILLED)  # line 314
        return True, avg_point

    # If there are no intersection points, return False and an empty list
    return False, []


def predictBounce(point, radius, holes):
    """
    Predicts the color of a circle centered at `point` with `radius`,
    and whether it falls into any of the `holes`.
    """
    is_in_hole = False
    color = (0, 0, 200)  # default color is blue

    for hole in holes:
        if (point[0] - radius >= hole[0] and point[1] - radius >= hole[1] and
                point[0] + radius <= hole[2] and point[1] + radius <= hole[3]):
            is_in_hole = True
            color = (0, 200, 0)  # color is green if in a hole

    return color, is_in_hole


def predict_path(collision_point, ball_rect, paths, holes):  # 284
    # Calculate the center of the colored ball
    ball_center = [ball_rect[0] + ball_rect[2] // 2, ball_rect[1] + ball_rect[3] // 2]

    # Calculate the equation of the line from the collision point to the ball center
    m, n = line_equation(collision_point, [ball_center[0] + 1, ball_center[1] + 1])

    # Determine the initial x coordinate based on the position of the collision point
    last_x = 30 if collision_point[0] > ball_center[0] else 790
    # Check if the ball will hit any walls
    for _ in range(2):
        x = last_x
        y = int((m * x) + n)

        # Handle collisions with the top and bottom walls
        if y >= 390:
            y = 390
            x = int((y - n) / m)
        if y <= 60:
            y = 60
            x = int((y - n) / m)

        # Handle collisions with the side walls
        if 75 < y < 350 and x >= 765:
            x = 765
            y = int((m * x) + n)
            last_x = 30
        if 75 < y < 350 and x <= 35:
            x = 35
            y = int((m * x) + n)
            last_x = 765

        # Add the new position to the path
        paths.append([x, y])

        # Check if the ball falls into a hole
        color, in_hole = predictBounce(paths[-1], 12, holes)
        if in_hole:
            return paths, color, in_hole

        # If not, invert the slope to simulate the ball bouncing off the wall
        m = -m
        n = y - (m * x)

    return paths, color, in_hole


# Predict the outcome of a shot
def predict_shot(hit_point, cue_ball, colored_balls, holes):
    with contextlib.suppress(TypeError):
        # Calculate the line equation from the hit point to the cue ball
        m1, n1 = line_equation([hit_point[0], hit_point[1]],
                               [cue_ball[0] + cue_ball[2] // 2, cue_ball[1] + cue_ball[3] // 2])

        # Generate points along the path of the cue ball towards the colored ball
        points = []
        x_last = (colored_balls[0] + colored_balls[2] // 2)
        x1, y1 = x_last, int((m1 * x_last) + n1)
        step = 1 if x_last >= cue_ball[0] + cue_ball[2] // 2 else -1
        for x in range(cue_ball[0] + cue_ball[2] // 2, x_last, step):
            y = int((m1 * x) + n1)
            points.append([x, y])

        # Check for collisions with the colored ball and predict the paths
        for point in points:
            bbox = [point[0] - cue_ball[2] // 2, point[1] - cue_ball[3] // 2, point[0] + cue_ball[2] // 2,
                    point[1] + cue_ball[3] // 2, ]
            collision, collision_point = detect_collision(bbox, [colored_balls[0], colored_balls[1],
                                                                 colored_balls[0] + colored_balls[2],
                                                                 colored_balls[1] + colored_balls[3], colored_balls[4]])
            if collision:
                return _extracted_from_predict_shot_25(
                    colored_balls, collision_point, holes, cue_ball
                )


# TODO Rename this here and in `predict_shot`
def _extracted_from_predict_shot_25(colored_balls, collision_point, holes, cue_ball):
    # Predict the paths of the colored ball
    paths = [[colored_balls[0] + colored_balls[2] // 2, colored_balls[1] + colored_balls[3] // 2]]
    paths, color, in_hole = predict_path(collision_point, colored_balls, paths, holes)

    # Draw the predicted paths and collision point on the image
    drawResult(paths, color, in_hole, False)
    draw_dotted_line(imgCropped, (cue_ball[0] + cue_ball[2] // 2, cue_ball[1] + cue_ball[3] // 2),
                     (collision_point[0], collision_point[1]), (200, 200, 200))
    cv2.circle(imgCropped, (collision_point[0], collision_point[1]), 5, (200, 200, 200), cv2.FILLED)

    return {"prediction": in_hole, "paths": paths, "color": color}


# Initialize variables
cap = cv2.VideoCapture("resources/shots.mp4")
frameWidth = 960
frameHeight = 540
size = (frameWidth, frameHeight)

result = cv2.VideoWriter('resources/shotsProcessed.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, size)

holes = locate_holes()
hitPoints = []
averageRadius = []
lastSpot = []
prediction = True
possibleOutcomes = []
frameId = 1

shotIndex = 1

# Start program
while True:
    success, frame = cap.read()
    imgRaw = cv2.resize(frame, (frameWidth, frameHeight))
    imgCropped = imgRaw[10:460, 80:881]

    # Detect the objects
    stick = locate_stick(imgCropped)
    cueBall = findCueBall(imgCropped)
    coloredBalls = find_colored_balls(imgCropped)
    if stick and cueBall and coloredBalls:
        if not lastSpot:
            lastSpot.append([cueBall[0] + cueBall[2] // 2, cueBall[1] + cueBall[3] // 2])
        lastSpot.append([cueBall[0] + cueBall[2] // 2, cueBall[1] + cueBall[3] // 2])
        difference = lambda a, b: math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))
        if difference(lastSpot[-1], lastSpot[-2]) >= 2 or frameId >= 1160:
            prediction = False
            mostLikely = {}
            count = 0
            for outcome in possibleOutcomes:
                if outcome != None and possibleOutcomes.count(outcome) > count:
                    count = possibleOutcomes.count(outcome)
                    mostLikely = outcome

            drawResult(mostLikely['paths'], mostLikely['color'], mostLikely['prediction'], True,
                       (count / len(possibleOutcomes)) * 1)



        elif len(lastSpot) > 2:
            if difference(lastSpot[-2], lastSpot[-3]) >= 2 and difference(lastSpot[-1], lastSpot[-2]) < 2:
                prediction = True
                hitPoints = []
                possibleOutcomes = []
                shotIndex += 1

        if prediction:
            hitPoint = get_hit_point(stick, cueBall, averageRadius, hitPoints)
            final = predict_shot(hitPoint, cueBall, coloredBalls, holes)
            possibleOutcomes.append(final)
    elif not prediction:
        drawResult(mostLikely['paths'], mostLikely['color'], mostLikely['prediction'], False,
                   (count / len(possibleOutcomes)) * 1)

    frameId += 1
    cv2.imshow("Result", imgRaw)
    result.write(imgRaw)
    if cv2.waitKey(75) & 0xFF == ord('q'):
        break

result.release()
