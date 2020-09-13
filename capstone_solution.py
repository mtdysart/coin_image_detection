import numpy as np
import cv2 as cv

img = cv.imread("capstone_coins.png", cv.IMREAD_GRAYSCALE)
original_image = cv.imread("capstone_coins.png", 1)
img = cv.GaussianBlur(img, (5, 5), 0)
# cv.namedWindow("Detected Coins", cv.WINDOW_KEEPRATIO)
# cv.imshow("Detected Coins", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Pixel height of the image
num_rows = img.shape[0]

# Get circle locations and radii using Circle Hough Transform 
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, num_rows / 4, param1=40, param2=50, minRadius=100, maxRadius=500)
print(circles)

if circles is not None:
    circles = np.uint16(np.around(circles))

    for circle in circles[0, :]:
        # First index is x coordinate, second is y
        center = (circle[0], circle[1])

        # Draw circle center
        cv.circle(img, center, 1, (0, 100, 100), 3)
        # Draw actual circle detected
        cv.circle(img, center, circle[2], (255, 0, 255), 3)

    # Display circles
    display_img = cv.resize(img, (1250, 580))
    
    cv.imshow("Detected coins", display_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

else:
    print("No circles were detected.")
