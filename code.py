# This code was made using Google Colab.
# The data was labeled manually, so this could result in errors. 
# You can use datasets already on Kaggle or Roboflow, up to you.

# Import dataset from roboflow
from roboflow import Roboflow
# Private API Key
rf = Roboflow(api_key = "use_your_API_Key")
# Workspace and project name (uploaded to roboflow)
project = rf.workspace("change_to_your_workspace").project("change_to_your_project_name")
version = project.version(1) # Change according to your project version 
dataset = version.download("yolov11")

# Import needed libraries
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random
import glob
import torch
from ultralytics import YOLO

# The answer to the great question of “life, the universe and everything” is 42.
np.random.seed(42)

# Define class nammes and colors of the polyogn
class_names = ["tumor"]
colors = np.random.uniform(0, 255, size=(len(class_names),3))

# Image preproccesing. This can be done automatically in roboflow if you prefer

# Apply blur
def apply_blur(img, kernel_size=5):
    return cv.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Enhance contrast
def enhance_contrast(img):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)  # Espacio de color LAB
    l, a, b = cv.split(lab)
    l = cv.equalizeHist(l)  # Ecualización de la luminosidad
    lab = cv.merge((l, a, b))
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)

# Apply histogram streching
def histogram_stretching(img):
  img_min = np.min(img)
  img_max = np.max(img)
  stretched = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
  return stretched

# Apply de CLAHE
def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    #clipLimit = controla amplificacion del contraste
    #tileGridSize 0 tamaño de cuadrícula
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab = cv.merge((l, a, b))
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)

# Edge detection
def detect_edges(img, low_threshold=30, high_threshold=150):
    return cv.Canny(img, low_threshold, high_threshold)

# Image segmentation

# Draw segmentation and class labels
def plot_segmentation(img, polygons, labels):
    h, w, _ = img.shape

    for polygon_num, polygon in enumerate(polygons):
        class_name = class_names[int(labels[polygon_num])]
        color = colors[int(labels[polygon_num])]

        points = []
        for i in range(0, len(polygon), 2):
            x = int(float(polygon[i]) * w)
            y = int(float(polygon[i + 1]) * h)
            points.append([x, y])
        points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

        cv.polylines(img, [points], isClosed=True, color=color, thickness=2)
        cv.fillPoly(img, [points], color=color)

        centroid_x = int(np.mean(points[:, 0, 0]))
        centroid_y = int(np.mean(points[:, 0, 1]))
        font_scale = 0.5
        font_thickness = 1
        cv.putText(img, class_name, (centroid_x, centroid_y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
      
    return img

# Upload, process and segment images 
def plot(image_paths, label_paths, num_samples):
  all_images = sorted(glob.glob(image_paths))
  all_labels = sorted(glob.glob(label_paths))

  if not all_images or not all_labels:
    print("File or label not found. Check path.")
    return

  num_images = len(all_images)
  plt.figure(figsize=(15,12))

  for i in range(num_samples):
    idx = random.randint(0, num_images-1)
    img = cv.imread(all_images[idx])

    if img is None:
      print(f"Error: Image could not be uploaded {all_images[idx]}.")
      continue

    # Apply image pre-processing and filters
    img_blurred = apply_blur(img)
    img_contrast = enhance_contrast(img_blurred)
    img_histogram_stretched = histogram_stretching(img_contrast)
    img_clahe = apply_clahe(img_histogram_stretched)

    polygons = []
    labels = []

    with open(all_labels[idx],'r') as f:
      for line in f.readlines():
        elements = line.split()
        label = int(elements[0])
        polygon_points = elements[1:] 
        polygons.append(polygon_points)
        labels.append(label)

    result_image = plot_segmentation(img_clahe, polygons, labels)

    # Show the results in different stages of processing
    plt.subplot(num_samples, 5, i * 5 + 1); plt.imshow(img[:, :, ::-1]); plt.title("Original"); plt.axis("off")
    plt.subplot(num_samples, 5, i * 5 + 2); plt.imshow(img_blurred[:, :, ::-1]); plt.title("Suavizado"); plt.axis("off")
    plt.subplot(num_samples, 5, i * 5 + 3); plt.imshow(img_contrast[:, :, ::-1]); plt.title("Contraste"); plt.axis("off")
    plt.subplot(num_samples, 5, i * 5 + 4); plt.imshow(img_histogram_stretched, cmap='gray'); plt.title("Histograma"); plt.axis("off")
    plt.subplot(num_samples, 5, i * 5 + 5); plt.imshow(result_image[:, :, ::-1]); plt.title("Segmentación"); plt.axis("off")

# Let's go and apply YOLO v11 for image segmentation
model = YOLO("yolo11n-seg.pt")

# Model trainning (you can change this as you'd find better for your own ML app)
train_results = model.train(
    data="use_your_path",  # path to dataset YAML
    epochs=100,  
    imgsz=416, 
    batch=64,
    conf=0.3
  )

# Test the model
model = YOLO('use_your_path') # File you're looking for is best.pt (usually on weights folder)
results = model(source='change_to_your_image_path.jpg', conf=0.30, save=True)
results[0].show()
