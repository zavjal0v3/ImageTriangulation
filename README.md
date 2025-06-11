# Image Triangulation with Color Quantization

A Python program that processes images by applying color quantization using KMeans clustering and then stylizes the image with Delaunay triangulation. The program detects feature points (corners) in the image, builds a triangulation mesh, and fills each triangle with the average color of its vertices, creating a polygonal art effect.

## Requirements

- Install Ubuntu latest
- Install [Python](https://www.python.org/downloads/) 3.6 or higher

## Install dependencies
    pip install opencv-python numpy scipy scikit-learn matplotlib tkinter
## Run production
    python ImageTriangulation.py


