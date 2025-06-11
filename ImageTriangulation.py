import cv2
import numpy as np
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def load_image_dialog():
    Tk().withdraw()
    filename = askopenfilename(title="Choose image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if not filename:
        print("No file selected.")
        return None
    img = cv2.imread(filename)
    if img is None:
        print("Failed to load image.")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(path, img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)

def kmeans_color_quantization(image, k=8):
    data = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    new_colors = kmeans.cluster_centers_.astype('uint8')
    labels = kmeans.labels_
    quantized = new_colors[labels].reshape(image.shape)
    return quantized

def detect_corners(image, max_corners, quality_level, min_distance):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    corners = corners.astype(np.int32).reshape(-1, 2)
    return corners

def average_color_in_triangle(image, triangle):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, triangle, 1)
    mean_color = cv2.mean(image, mask=mask)[:3]
    return np.array(mean_color, dtype=np.uint8)

def triangulate_and_color(image, points):
    tri = Delaunay(points)
    result = np.zeros_like(image)
    for simplex in tri.simplices:
        triangle = points[simplex]
        color = average_color_in_triangle(image, triangle)
        cv2.fillConvexPoly(result, triangle, color.tolist())
    return result, tri

def calculate_ase(original, processed):
    diff = np.abs(original.astype(np.int16) - processed.astype(np.int16))
    ase = np.mean(diff)
    return ase

def get_user_input(prompt, default, cast_func):
    user_input = input(f"{prompt} (default {default}): ").strip()
    if not user_input:
        return default
    try:
        return cast_func(user_input)
    except ValueError:
        print("Invalid input, using default.")
        return default

if __name__ == "__main__":
    img = load_image_dialog()
    if img is None:
        exit()

    max_corners = get_user_input("Enter max_corners", 1500, int)
    quality_level = get_user_input("Enter quality_level", 0.001, float)
    min_distance = get_user_input("Enter min_distance", 3, int)

    quantized = kmeans_color_quantization(img, k=8)
    corners = detect_corners(quantized, max_corners=max_corners, quality_level=quality_level, min_distance=min_distance)
    processed, tri = triangulate_and_color(quantized, corners)
    processed_with_edges = processed

    ase = calculate_ase(img, processed)
    print(f'Average Squared Error (ASE): {ase:.2f}')

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title('Original')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title('Triangulated')
    plt.imshow(processed_with_edges)
    plt.axis('off')

    plt.show()

    save_image('output.png', processed_with_edges)
