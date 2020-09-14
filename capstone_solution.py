import numpy as np
import cv2 as cv

def classify_coin(radius, brightness):
    """
    Returns the value of a coin based on its size and brightness, ie, returned value of 5 => 5p coin
    """

    # Check coin brightness to determine if it is silver or copper (higher brightness is silver)
    if brightness > 170:
        return 10 if radius > 130 else 5

    else:
        if radius >= 80 and radius < 145:
            return 1
        
        elif radius >= 145 and radius < 200:
            return 2

    return 0

def find_50p_coin(val):
    """
    Finds and draws the non-circular 50 p coin and adds 50 to the current total value of the coins
    """
    # Convert image to black and white to find contours
    x, threshold = cv.threshold(img, 108, 255, cv.THRESH_BINARY) 

    # Detect shapes in image by selecting region with same intensity
    contours, hierarchy = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2:]

    # Filter the contours to those with area > 400
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > 400] 

    # The contour for the 50 pence coin is in index 7
    # Approximate the polynomial curve of the coin and draw it
    approx = cv.approxPolyDP(contours[7], 0.018 * cv.arcLength(contours[7], True), True)
    cv.drawContours(original_image, [approx], 0, (255, 0, 255), 5) 
    cv.putText(original_image, "50p", (1180, 500), cv.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)

    return val + 50

value_sum = 0
img = cv.imread("capstone_coins.png", cv.IMREAD_GRAYSCALE)
original_image = cv.imread("capstone_coins.png", 1)
img = cv.GaussianBlur(img, (5, 5), 0)
num_rows = img.shape[0]

# Get circle locations and radii using Circle Hough Transform 
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, num_rows / 4, param1=40, param2=50, minRadius=100, maxRadius=500)

if circles is not None:
    circles = np.uint16(np.around(circles))

    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        
        # Get brigtness value at center
        bright_val = img[circle[1], circle[0]]

        # Get coin value and add it to the total
        coin_value = classify_coin(circle[2], bright_val)
        value_sum += coin_value

        # Draw the circles and print coin value
        cv.circle(original_image, center, 1, (0, 100, 100), 3)
        cv.circle(original_image, center, circle[2], (255, 0, 255), 3)

        val_str = "{}p".format(coin_value)
        cv.putText(original_image, val_str, (center[0] + 20, center[1]), cv.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)

else:
    print("No coins were detected.")


total_value = find_50p_coin(value_sum)

# Print total value
total_str = "The total value is {}p".format(total_value)
cv.putText(original_image, total_str, (100, 100), cv.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)

# Resize img to display
display_img = cv.resize(original_image, (1250, 580))

cv.imshow("Detected coins", display_img)
k = cv.waitKey(0)
if k == ord("s"):
    cv.imwrite("result.png", original_image)

cv.destroyAllWindows()
