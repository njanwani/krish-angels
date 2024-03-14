import cv2
import numpy as np

# Define a sample contour
contour = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])
print(contour.shape)

# Create an image to draw the contour on
image = np.zeros((400, 400), dtype=np.uint8)
cv2.drawContours(image, [contour], 0, 255, -1)

# Define a point to check
point = (200, 200)

# Check if the point is inside the contour
distance = cv2.pointPolygonTest(contour, point, False)

if distance > 0:
    print("Point is inside contour.")
elif distance == 0:
    print("Point is on contour.")
else:
    print("Point is outside contour.")

# Display the image with the contour and point
cv2.circle(image, point, 5, 128, -1)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
