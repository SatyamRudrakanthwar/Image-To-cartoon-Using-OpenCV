import cv2
import numpy as np
from sklearn.cluster import KMeans  

def quantize_colors(image, num_colors=64):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    pixels = lab_image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    quantized_colors = kmeans.cluster_centers_.astype(np.uint8)
    quantized_image = quantized_colors[kmeans.labels_].reshape(image.shape)
    return cv2.cvtColor(quantized_image, cv2.COLOR_Lab2BGR)

img_rgb = cv2.imread("nobi.webp")

img_quantized = quantize_colors(img_rgb, num_colors=32)

num_down = 2
num_bilateral = 7

print(img_rgb.shape)

img_rgb = cv2.resize(img_rgb, (800, 800))
img_rgb = cv2.resize(img_rgb, (800, 880))

img_color = img_rgb
for _ in range(num_down):
    img_color = cv2.pyrDown(img_color)

for _ in range(num_bilateral):
    img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

for _ in range(num_down):
    img_color = cv2.pyrUp(img_color)

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
img_blur = cv2.medianBlur(img_gray, 7)

img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)

img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
img_cartoon = cv2.bitwise_and(img_color, img_edge)


stack = np.hstack([img_rgb, img_cartoon])
cv2.imshow("Stacked Image", stack)

cv2.waitKey(0)
cv2.destroyAllWindows()
