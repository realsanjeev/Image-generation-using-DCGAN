# /// script
# dependencies = [
#   "numpy",
#   "matplotlib",
#   "scikit-learn",
#   "scikit-image",
# ]
# ///

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from skimage.segmentation import slic, mark_boundaries
from skimage import color, io, graph
from skimage.util import img_as_float, img_as_ubyte

def kmeans_segmentation(image, k=3):
    """
    Perform K-Means based image segmentation.
    """
    # Ensure float for clustering, but keep original dtype for output
    img = img_as_float(image)
    img_flat = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(img_flat)
    centers = kmeans.cluster_centers_  # in [0,1]

    segmented = centers[labels].reshape(image.shape)
    return img_as_ubyte(segmented)


def meanshift_segmentation(image):
    """
    Perform Mean-Shift based image segmentation.
    """
    img = img_as_float(image)
    img_flat = img.reshape((-1, 3))

    # Estimate bandwidth on subset for speed
    bandwidth = estimate_bandwidth(img_flat, quantile=0.1, n_samples=500)

    if bandwidth == 0:
        bandwidth = 0.1  # fallback to avoid failure

    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = meanshift.fit_predict(img_flat)

    segmented = meanshift.cluster_centers_[labels]
    segmented = segmented.reshape(image.shape)
    return img_as_ubyte(segmented)


def normalized_cut_segmentation(image):
    """
    Perform Normalized Graph Cut Segmentation using superpixels.
    Uses SLIC + RAG + normalized cut.
    """
    # Work in float
    img_float = img_as_float(image)
    img_lab = color.rgb2lab(img_float)

    # Generate superpixels
    segments = slic(img_lab, n_segments=300, compactness=30, sigma=1, start_label=0)

    # Build region adjacency graph using mean color in Lab space
    rag = graph.rag_mean_color(img_lab, segments, mode='similarity')

    # Apply normalized cut — returns *new* label map (same shape as segments)
    labels = graph.cut_normalized(segments, rag)

    # Ensure labels are contiguous integers starting from 0
    # (cut_normalized may leave gaps or negative values in older versions)
    labels = labels.astype(int)
    uniq = np.unique(labels)
    label_map = {old: new for new, old in enumerate(uniq)}
    labels = np.vectorize(label_map.get)(labels)

    # Color segments using original image (RGB)
    segmented = color.label2rgb(labels, image, kind='avg', bg_label=None)
    return img_as_ubyte(segmented)


def main(image_path):
    image = io.imread(image_path)

    # Preprocess: ensure RGB and uint8
    if image.ndim == 2:  # Grayscale
        image = color.gray2rgb(image)
    elif image.ndim == 3:
        if image.shape[2] == 4:  # RGBA
            image = color.rgba2rgb(image)
        # Ensure uint8 for consistency
        if image.dtype != np.uint8:
            image = img_as_ubyte(image)
    else:
        raise ValueError("Unsupported image format")

    try:
        print(f"[INFO]: Performing K-Means Segmentation")
        kmeans_img = kmeans_segmentation(image, k=4)
    except Exception as e:
        print(f"[WARN] K-Means failed: {e}")
        kmeans_img = image

    try:
        print(f"[INFO]: Performing Mean-Shift Segmentation")
        meanshift_img = meanshift_segmentation(image)
    except Exception as e:
        print(f"[WARN] Mean-Shift failed: {e}")
        meanshift_img = image

    try:
        print(f"[INFO]: Performing Normalized Cut Segmentation")
        graphcut_img = normalized_cut_segmentation(image)
    except Exception as e:
        print(f"[WARN] Normalized Cut failed: {e}")
        graphcut_img = image

    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(kmeans_img)
    axes[1].set_title("K-Means Segmentation")
    axes[1].axis("off")

    axes[2].imshow(meanshift_img)
    axes[2].set_title("Mean-Shift Segmentation")
    axes[2].axis("off")

    axes[3].imshow(graphcut_img)
    axes[3].set_title("Normalized Graph Cut")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig("sk_segmentation.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("[ERROR]: There must be exactly one argument")
        print("[USAGE]: python segmentation_demo.py <image_path>")
        sys.exit(1)

    main(sys.argv[1])