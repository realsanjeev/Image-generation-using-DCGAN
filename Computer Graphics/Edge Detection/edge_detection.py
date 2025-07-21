"""
Edge and Corner Detection Algorithms in Computer Vision.

This script demonstrates several classical edge and corner detection techniques used in image processing:

1. Sobel Edge Detection: Computes gradient magnitude using Sobel operators to highlight edges.
2. Prewitt Edge Detection: Similar to Sobel but uses simpler convolution kernels.
3. Roberts Edge Detection: Detects edges using diagonal gradient operators.
4. Laplacian of Gaussian: Smooths the image with Gaussian blur and detects edges with Laplacian filter.
5. Canny Edge Detection: A multi-stage algorithm to detect a wide range of edges in images.
6. Harris Corner Detection: Identifies corners by analyzing local gradients and variations in intensity.

The script fetches an image from a URL, applies each method, and displays the results.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import urllib.request

def show(title, image, cmap="gray"):
    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.show()

def sobel_edge_detection(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.uint8(np.clip(magnitude, 0, 255))

def prewitt_edge_detection(image):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1,  1], [0, 0,  0], [-1, -1, -1]])
    x = convolve(image.astype(np.float64), kernelx)
    y = convolve(image.astype(np.float64), kernely)
    magnitude = np.sqrt(x**2 + y**2)
    return np.uint8(np.clip(magnitude, 0, 255))

def roberts_edge_detection(image):
    kernelx = np.array([[1, 0], [0, -1]])
    kernely = np.array([[0, 1], [-1, 0]])
    x = convolve(image.astype(np.float64), kernelx)
    y = convolve(image.astype(np.float64), kernely)
    magnitude = np.sqrt(x**2 + y**2)
    return np.uint8(np.clip(magnitude, 0, 255))

def laplacian_of_gaussian(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    return np.uint8(np.clip(np.abs(laplacian), 0, 255))

def canny_edge_detection(image, low_threshold=100, high_threshold=200):
    return cv2.Canny(image, low_threshold, high_threshold)

def harris_corner_detection(image):
    gray = np.float32(image)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    result = image.copy()
    result[dst > 0.01 * dst.max()] = 255
    return result

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    return image

def main():
    path = "https://thumbs.dreamstime.com/b/beautiful-landscape-valley-alpine-mountains-small-houses-seefeld-rural-scene-majestic-picturesque-view-40712070.jpg"
    img = url_to_image(path)

    if img is None:
        print(f"[ERROR]: Image not found in path: {path}")
        return
    
    show("Original Image", img)

    sobel = sobel_edge_detection(img)
    show("Sobel Edge Detection", sobel)

    prewitt = prewitt_edge_detection(img)
    show("Prewitt Edge Detection", prewitt)

    roberts = roberts_edge_detection(img)
    show("Roberts Edge Detection", roberts)

    laplacian = laplacian_of_gaussian(img)
    show("Laplacian of Gaussian", laplacian)

    canny = canny_edge_detection(img)
    show("Canny Edge Detection", canny)

    harris = harris_corner_detection(img)
    show("Harris Corner Detection", harris)

if __name__ == "__main__":
    main()