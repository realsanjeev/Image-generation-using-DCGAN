# Segmentation

**Segmentation** is a fundamental task in image processing and computer vision that involves partitioning an image into meaningful, non-overlapping regions. Each region groups pixels that share similar properties such as color, intensity, texture, or spatial proximity. The primary goal of segmentation is to simplify or transform an image’s representation, making it easier to analyze, interpret, or use for higher-level tasks such as object detection, recognition, and tracking.

Segmentation techniques can be broadly categorized into **clustering-based**, **density-based**, and **graph-based** approaches.

In modern machine learning and deep learning, two additional segmentation types are commonly used: **instance segmentation** and **semantic segmentation**. These approaches will not be discussed here, but for readers who are interested:

* **Instance segmentation** distinguishes individual object instances within the same class.
* **Semantic segmentation** assigns a class label to every pixel in the image.

Below are three classical, algorithm-based segmentation techniques.


## 1. K-Means Clustering

**K-means clustering** is a clustering-based segmentation technique that models segmentation as an optimization problem. Pixels are grouped into *K* clusters such that the variance within each cluster is minimized.

### Objective function

K-means minimizes the **within-cluster sum of squared distances**:

$$\arg\min_{C} \sum_{i=1}^{K} \sum_{x \in C_i} | x - \mu_i |^2$$

where:

* $C_i$ is the *i*-th cluster
* $x$ is a pixel feature vector
* $\mu_i$ is the centroid of cluster $C_i$

### How it works

1. Select the number of clusters (*K*).
2. Initialize *K* cluster centroids.
3. Assign each pixel to the nearest centroid (usually using Euclidean distance).
4. Update centroids as the mean of assigned pixels.
5. Repeat until convergence.

### Advantages

* Simple and easy to implement
* Computationally efficient
* Effective when regions are well separated

### Limitations

* Requires predefined *K*
* Sensitive to initialization
* Assumes spherical, equally sized clusters
* Does not enforce spatial continuity


## 2. Mean-Shift Segmentation

**Mean-shift segmentation** is a density-based technique that treats pixels as samples from a probability density function and seeks regions of high density (modes).

### Core idea

Given a point (x), the **mean-shift vector** is computed as:

$$m(x) = \frac{\sum_{i} K(x_i - x) , x_i}{\sum_{i} K(x_i - x)} - x$$

where:

* $x_i$ are neighboring points
* $K(\cdot)$ is a kernel function (e.g., Gaussian)

The algorithm iteratively shifts (x) toward the direction of maximum density.

### How it works

1. Represent each pixel in a joint feature space (color + spatial coordinates).
2. Place a kernel window around the pixel.
3. Compute the mean of points inside the window.
4. Shift the window toward the mean.
5. Repeat until convergence.
6. Pixels converging to the same mode form a segment.

### Key parameter

* **Bandwidth ((h))**: Controls kernel size and segmentation scale.

### Advantages

* No need to specify the number of segments
* Handles arbitrarily shaped regions
* Good edge preservation

### Limitations

* Computationally expensive
* Sensitive to bandwidth selection
* Slower than k-means


## 3. SLIC (Simple Linear Iterative Clustering)

**SLIC** is an algorithm that adapts k-means clustering to generate **superpixels**—groups of connected pixels with similar colors. It performs clustering in a 5D space defined by the ($L, a, b$) color space and $(x, y)$ pixel coordinates.

### Core idea

SLIC limits the search space for each cluster center to a region proportional to the superpixel size ($S \times S$), making it significantly faster than standard k-means.

The distance measure $D$ combines color distance $d_c$ and spatial distance $d_s$:

$$D = \sqrt{d_c^2 + \left(\frac{m}{S}\right)^2 d_s^2}$$

where:
* $m$ controls the compactness of the superpixels (weighting spatial vs. color distance).
* $S$ is the grid interval (approximate superpixel size).

### Advantages

* **Fast and memory efficient** ($O(N)$ complexity)
* **Adheres well to boundaries**
* Generates regular, compact superpixels

### Limitations

* Adherence to boundaries depends on the compactness factor $m$
* Not a full segmentation method itself (usually a preprocessing step)


## 4. Normalized Graph Cut Segmentation

**Normalized Graph Cut (N-Cut)** is a graph-based method where an image is represented as a weighted graph and segmentation is performed by partitioning the graph.

### Graph formulation

* Nodes represent pixels (or superpixels)
* Edge weights represent similarity between pixels

The **normalized cut** criterion is defined as:

$$\text{Ncut}(A,B) = \frac{\text{cut}(A,B)}{\text{assoc}(A,V)} + \frac{\text{cut}(A,B)}{\text{assoc}(B,V)}$$

where:

* $\text{cut}(A,B)$ measures dissimilarity between regions (A) and (B)
* $\text{assoc}(A,V)$ measures total connectivity of region (A)

### Key idea

The goal is to minimize Ncut, which simultaneously:

* Separates dissimilar regions
* Avoids creating very small segments

The optimization is solved using **eigenvalue decomposition**.

### Advantages

* High-quality, globally consistent segmentation
* Preserves object boundaries
* Balances region size and similarity

### Limitations

* Computationally expensive
* Requires careful similarity design
* Not suitable for real-time large images


## Summary

| Technique            | Type             | Mathematical Goal                | Main Strength             | Main Limitation           |
| -------------------- | ---------------- | -------------------------------- | ------------------------- | ------------------------- |
| K-Means              | Clustering-based | Minimize intra-cluster variance  | Simple and fast           | Requires predefined *K*   |
| Mean-Shift           | Density-based    | Find modes of a density function | No need for *K*           | High computational cost   |
| SLIC                 | Superpixel       | Local K-Means in 5D space        | Fast, boundary adhering   | Parameter sensitivity     |
| Normalized Graph Cut | Graph-based      | Minimize normalized graph cut    | High-quality segmentation | Computationally intensive |

**References**

* *K-Means Clustering* — Wikipedia: [https://en.wikipedia.org/wiki/K-means_clustering](https://en.wikipedia.org/wiki/K-means_clustering)
* *Mean Shift* — Wikipedia: [https://en.wikipedia.org/wiki/Mean_shift](https://en.wikipedia.org/wiki/Mean_shift)
* *`skimage.segmentation`* — scikit-image Documentation: [https://scikit-image.org/docs/stable/api/skimage.segmentation.html](https://scikit-image.org/docs/stable/api/skimage.segmentation.html)
* *Graph Cuts in Computer Vision* — Wikipedia: [https://en.wikipedia.org/wiki/Graph_cuts_in_computer_vision](https://en.wikipedia.org/wiki/Graph_cuts_in_computer_vision)


## Implementation from Scratch

A full implementation of these algorithms from scratch (using only `numpy`, without `scikit-image` or `opencv` for core logic) can be found in `segmentation_scratch.py`.

This script includes:
- **Bilinear Resize**: A robust manual implementation of image resizing.
- **K-Means**: Standard implementation with improved initialization.
- **Mean-Shift**: Optimized with spatial indexing for better performance.
- **SLIC**: Simple Linear Iterative Clustering for superpixel generation.
- **Normalized Cut**: Full spectral clustering implementation using SLIC superpixels as nodes.

### Usage

```bash
python segmentation_scratch.py <path_to_image>
```

Example:

```bash
python segmentation_scratch.py sample.jpg
``` 


