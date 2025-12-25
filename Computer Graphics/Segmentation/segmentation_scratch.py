import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import time
def load_image(path):
    """Load image as float32 numpy array (H, W, 3) in [0,255]."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32)

def bilinear_resize(image, new_shape):
    """
    Resize image to new_shape (new_h, new_w) using bilinear interpolation.
    """
    h, w, c = image.shape
    new_h, new_w = new_shape

    # Create coordinate grids for the new image
    y = np.linspace(0, h-1, new_h)
    x = np.linspace(0, w-1, new_w)
    xv, yv = np.meshgrid(x, y)

    # Floor indices
    x_low = np.floor(xv).astype(int)
    y_low = np.floor(yv).astype(int)
    
    # Ceiling indices (x_low + 1)
    x_high = x_low + 1
    y_high = y_low + 1

    # Clamp indices
    x_low_clamped = np.clip(x_low, 0, w - 1)
    x_high_clamped = np.clip(x_high, 0, w - 1)
    y_low_clamped = np.clip(y_low, 0, h - 1)
    y_high_clamped = np.clip(y_high, 0, h - 1)

    # Interpolation weights
    # dx, dy relative to the low index
    dx = xv - x_low
    dy = yv - y_low

    wa = (1 - dx) * (1 - dy)
    wb = dx * (1 - dy)
    wc = (1 - dx) * dy
    wd = dx * dy

    # Sample the 4 pixels
    ia = image[y_low_clamped, x_low_clamped]
    ib = image[y_low_clamped, x_high_clamped]
    ic = image[y_high_clamped, x_low_clamped]
    id_ = image[y_high_clamped, x_high_clamped]

    # Combine
    resized = (ia * wa[:,:,None] + ib * wb[:,:,None] + 
               ic * wc[:,:,None] + id_ * wd[:,:,None])
    return resized

def nearest_neighbor_resize(image, new_shape):
    """
    Resize image to new_shape (new_h, new_w) using nearest neighbor.
    Better for labels/segmentation masks.
    """
    h, w, c = image.shape
    new_h, new_w = new_shape

    y = (np.linspace(0, h - 1, new_h) + 0.5).astype(int)
    x = (np.linspace(0, w - 1, new_w) + 0.5).astype(int)
    y = np.clip(y, 0, h - 1)
    x = np.clip(x, 0, w - 1)

    return image[y[:, None], x]

def fast_squared_distances(X, C):
    """
    Compute squared Euclidean distance between X (N, D) and C (K, D).
    Returns (N, K) matrix.
    Using identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.dot(b.T)
    """
    # X_sq: (N, 1), C_sq: (1, K)
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    C_sq = np.sum(C**2, axis=1, keepdims=True).T
    dist_sq = X_sq + C_sq - 2 * np.dot(X, C.T)
    # Handle precision issues (ensure >= 0)
    return np.maximum(dist_sq, 0)


def kmeans_segmentation(image, K=3, max_iter=20, tol=1e-4, seed=42):
    np.random.seed(seed)
    h, w, c = image.shape
    X = image.reshape(-1, c)

    # --- K-Means++ initialization ---
    centroids = np.empty((K, c), dtype=np.float32)
    centroids[0] = X[np.random.randint(len(X))]

    for k in range(1, K):
        # Optimized distance for initialization
        dist_sq = np.min(fast_squared_distances(X, centroids[:k]), axis=1)
        total = dist_sq.sum()
        if total == 0:
            centroids[k] = X[np.random.randint(len(X))]
        else:
            probs = dist_sq / total
            centroids[k] = X[np.random.choice(len(X), p=probs)]
    
    # --- Lloyd iterations ---
    for i in range(max_iter):
        distances = fast_squared_distances(X, centroids)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.empty_like(centroids)
        for k in range(K):
            mask = labels == k
            if np.any(mask):
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                new_centroids[k] = centroids[k]
        
        shift = np.linalg.norm(centroids - new_centroids)
        centroids = new_centroids
        if shift < tol:
            break

    segmented = centroids[labels].reshape(h, w, c)
    return segmented.clip(0, 255).astype(np.uint8)



def meanshift_segmentation(
    image,
    bandwidth=30,
    max_iters=10,
    convergence_tol=1.0,
    chunk_size=1000,
):
    h, w, c = image.shape
    N = h * w

    # Feature construction: (R, G, B, X, Y)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    features = np.zeros((N, 5), dtype=np.float32)
    features[:, :3] = image.reshape(N, 3)
    features[:, 3] = xs.ravel()
    features[:, 4] = ys.ravel()

    # Scale spatial features relative to bandwidth
    # This helps balance color vs position importance
    spatial_scale = bandwidth / max(h, w)
    features[:, 3:] *= spatial_scale

    points = features.copy()
    active = np.ones(N, dtype=bool)

    # Iteration
    for it in range(max_iters):
        if not np.any(active):
            break

        new_points = points.copy()

        # We only update active points to save time
        active_indices = np.where(active)[0]

        for start in range(0, len(active_indices), chunk_size):
            end = min(start + chunk_size, len(active_indices))
            idx_chunk = active_indices[start:end]
            query = points[idx_chunk]

            # Fast squared distance
            dist_sq = fast_squared_distances(query, features)
            mask = dist_sq < bandwidth**2

            # Weighted average (flat kernel)
            for i, q_idx in enumerate(idx_chunk):
                neighbors = features[mask[i]]
                if len(neighbors) > 0:
                    new_pos = neighbors.mean(axis=0)
                    shift = np.linalg.norm(new_pos - query[i])
                    new_points[q_idx] = new_pos
                    if shift < convergence_tol:
                        active[q_idx] = False
                else:
                    active[q_idx] = False
        points = new_points
        if not np.any(active):
            break

    # Cluster modes
    labels = np.full(N, -1, dtype=np.int32)
    modes = []

    # Faster clustering
    for i in range(N):
        if labels[i] != -1:
            continue

        mode = points[i]
        # Find all points close to this mode
        dist_sq = np.sum((points - mode)**2, axis=1)
        matches = dist_sq < (bandwidth / 2)**2

        # Only label points that haven't been assigned yet
        unlabeled_matches = matches & (labels == -1)
        labels[unlabeled_matches] = len(modes)
        modes.append(mode[:3])

    modes = np.array(modes)
    segmented = modes[labels].reshape(h, w, 3)
    return segmented.clip(0, 255).astype(np.uint8)


def slic_segmentation(image, n_segments=100, compactness=10, max_iter=10, return_labels=False):
    """
    Optimized SLIC implementation from scratch.
    """
    h, w, c = image.shape
    N = h * w
    S = int(np.sqrt(N / n_segments))

    # Initialize centers on a grid
    xs = np.arange(S // 2, w, S)
    ys = np.arange(S // 2, h, S)
    xv, yv = np.meshgrid(xs, ys)

    centers = np.zeros((xv.size, 5), dtype=np.float32)
    centers[:, 3] = xv.ravel()
    centers[:, 4] = yv.ravel()
    for i in range(len(centers)):
        centers[i, :3] = image[int(centers[i, 4]), int(centers[i, 3])]

    labels = np.full((h, w), -1, dtype=np.int32)
    distances = np.full((h, w), np.inf, dtype=np.float32)

    # Normalization factors
    M = compactness

    for _ in range(max_iter):
        distances.fill(np.inf)
        for i in range(len(centers)):
            cx, cy = int(centers[i, 3]), int(centers[i, 4])

            # Search region [25 X 25]
            y_min, y_max = max(0, cy - S), min(h, cy + S)
            x_min, x_max = max(0, cx - S), min(w, cx + S)

            roi = image[y_min:y_max, x_min:x_max]

            # Color distance
            d_color_sq = np.sum((roi - centers[i, :3])**2, axis=2)

            # Spatial distance
            yy, xx = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')
            d_spatial_sq = (xx - centers[i, 3])**2 + (yy - centers[i, 4])**2

            # SLIC Distance: D = sqrt( d_c^2 + (M/S)^2 * d_s^2 )
            d = np.sqrt(d_color_sq + (M/S)**2 * d_spatial_sq)

            mask = d < distances[y_min:y_max, x_min:x_max]
            distances[y_min:y_max, x_min:x_max][mask] = d[mask]
            labels[y_min:y_max, x_min:x_max][mask] = i

        # Update centers
        for i in range(len(centers)):
            mask = (labels == i)
            if np.any(mask):
                centers[i, :3] = image[mask].mean(axis=0)
                coords = np.where(mask)
                centers[i, 4] = coords[0].mean()
                centers[i, 3] = coords[1].mean()

    if return_labels:
        return labels, centers[:, :3]
    
    # Color segments by mean color
    segmented = np.zeros_like(image)
    for i in range(len(centers)):
        mask = (labels == i)
        segmented[mask] = centers[i, :3]

    return segmented.clip(0, 255).astype(np.uint8)

def ncut_segmentation(image, n_segments=100, compactness=10, sigma=20.0):
    """
    Perform Normalized Cut on superpixels.
    Nodes = Superpixels (SLIC)
    Edges = Color similarity between adjacent superpixels.
    """
    h, w, c = image.shape

    # 1. Get superpixels
    labels, node_colors = slic_segmentation(
        image, n_segments=n_segments, compactness=compactness,
        max_iter=10, return_labels=True
    )
    
    # 2. Filter unused nodes (isolated superpixels)
    unique_labels = np.unique(labels)
    # Map old label -> new label
    label_map = np.full(len(node_colors), -1, dtype=int)
    label_map[unique_labels] = np.arange(len(unique_labels))
    
    # Remap image labels
    labels = label_map[labels]
    
    # Keep only used colors
    node_colors = node_colors[unique_labels]
    num_nodes = len(unique_labels)
    
    if num_nodes < 2:
        return np.zeros_like(image)

    # 3. Build Adjacency Matrix (W)
    W = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # Check vertical and horizontal adjacency
    adj_v = np.stack([labels[:-1, :], labels[1:, :]], axis=-1).reshape(-1, 2)
    adj_h = np.stack([labels[:, :-1], labels[:, 1:]], axis=-1).reshape(-1, 2)
    
    # Unique edges
    adj = np.unique(np.vstack([adj_v, adj_h]), axis=0)
    
    # Remove self-loops
    adj = adj[adj[:, 0] != adj[:, 1]]

    for i, j in adj:
        dist_sq = np.sum((node_colors[i] - node_colors[j])**2)
        weight = np.exp(-dist_sq / (2 * sigma**2))
        W[i, j] = W[j, i] = weight

    # Add small global connectivity to prevent disconnected components
    # This ensures 2nd eigenvector is meaningful (Fiedler vector)
    W += 1e-6

    # 4. Normalized Cut via Eigen-decomposition
    d = np.sum(W, axis=1)
    
    # Standard N-Cut: L_sym = I - D^-1/2 W D^-1/2
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    L_sym = np.eye(num_nodes) - D_inv_sqrt @ W @ D_inv_sqrt
    
    # Solve for eigenvalues/vectors
    evals, evecs = np.linalg.eigh(L_sym)

    # 2nd smallest eigenvector (Fiedler vector) is at index 1
    # eigenvectors are columns
    v = D_inv_sqrt @ evecs[:, 1]

    # Thresholding to binary cut
    cut_labels = (v > 0).astype(int)

    # Color the image based on the cut
    side0_pts = node_colors[cut_labels == 0]
    side1_pts = node_colors[cut_labels == 1]

    color0 = side0_pts.mean(axis=0) if len(side0_pts) > 0 else np.array([0,0,0])
    color1 = side1_pts.mean(axis=0) if len(side1_pts) > 0 else np.array([255,255,255])

    final_colors = np.stack([color0, color1])
    pixel_labels = cut_labels[labels]

    segmented = final_colors[pixel_labels]
    return segmented.clip(0, 255).astype(np.uint8)



def main(image_path):
    print(f"[INFO]: Loading Image: {image_path}")
    image = load_image(image_path)
    original_h, original_w = image.shape[:2]

    # Downsample for processing speed if necessary
    process_dim = 120
    if max(original_h, original_w) > process_dim:
        scale = process_dim / max(original_h, original_w)
        new_w, new_h = int(original_w * scale), int(original_h * scale)
        image_small = bilinear_resize(image, (new_h, new_w))
        print(f"[INFO]: Resized for processing: {new_w}X{new_h}")
    else:
        image_small = image

    results = {}
    
    # 1. K-Means
    print("Running Optimized K-Means...")
    start = time.time()
    results['K-Means'] = kmeans_segmentation(image_small, K=5)
    print(f"K-Means took {time.time() - start:.2f}s")

    # 2. Mean-Shift
    print("Running Optimized Mean-Shift...")
    start = time.time()
    # Using a smaller bandwidth or downsampled image for Mean-Shift as it's still heavy
    results['Mean-Shift'] = meanshift_segmentation(image_small, bandwidth=25)
    print(f"Mean-Shift took {time.time() - start:.2f}s")

    # 3. SLIC
    print("Running SLIC Segmentation...")
    start = time.time()
    results['SLIC'] = slic_segmentation(image_small, n_segments=100, compactness=20)
    print(f"SLIC took {time.time() - start:.2f}s")

    # 4. Normalized Cut
    print("Running Real Normalized Cut...")
    start = time.time()
    results['N-Cut'] = ncut_segmentation(image_small, n_segments=150, compactness=15)
    print(f"N-Cut took {time.time() - start:.2f}s")

    # Upsample results to original size
    final_results = {}
    for name, img in results.items():
        if img.shape[:2] != (original_h, original_w):
            print(f"Upsampling {name}...")
            final_results[name] = nearest_neighbor_resize(img, (original_h, original_w))
        else:
            final_results[name] = img

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    axes[0].imshow(image.astype(np.uint8))
    axes[0].set_title("Original Image", fontsize=14)
    
    # Track indices for subplots
    plot_idx = 1
    for name, img in final_results.items():
        axes[plot_idx].imshow(img)
        axes[plot_idx].set_title(f"{name} Segmentation", fontsize=14)
        plot_idx += 1

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    output_path = "output_segmentation_optimized.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Result saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python segmentation_scratch.py <image_path>")
        sys.exit(1)
    
    main(sys.argv[1])
