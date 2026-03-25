import cv2
import numpy as np

# ImageNet normalization constants matching DINOv2 preprocessing
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr, imgsz=256):
    """Resize and normalize a BGR image for DINOv2 inference.

    Replicates the AutoImageProcessor pipeline (resize, ImageNet normalize)
    without requiring the transformers library at inference time.

    Args:
        img_bgr: BGR image as a numpy array (H, W, 3), uint8.
        imgsz: int for aspect-ratio preserving resize (scales so the shortest
               side equals imgsz), or (h, w) tuple for exact resize.

    Returns:
        numpy.ndarray: Float32 array of shape (1, 3, H', W'), normalized.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if isinstance(imgsz, int):
        h, w = img_rgb.shape[:2]
        scale = imgsz / min(h, w)
        new_h = round(h * scale)
        new_w = round(w * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        target_h, target_w = imgsz
        img_resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    inp = img_resized.astype(np.float32) / 255.0
    inp = (inp - IMAGENET_MEAN) / IMAGENET_STD          # (H, W, 3)
    inp = np.transpose(inp, (2, 0, 1))[np.newaxis, :]   # (1, 3, H, W)
    return np.ascontiguousarray(inp, dtype=np.float32)


def reshape_patches(last_hidden_state, img_shape):
    """Reshape flat DINOv2 patch tokens into a spatial grid.

    Args:
        last_hidden_state: Tensor or array of shape (1, num_patches+1, D).
            The first token (CLS) is dropped.
        img_shape: (H, W) or (H, W, C) used to infer the patch grid aspect ratio.

    Returns:
        Reshaped tensor/array of shape (1, h, w, D) where h*w == num_patches.
    """
    patches = last_hidden_state[:, 1:, :]  # drop CLS token
    num_patches = patches.shape[1]
    H, W = img_shape[0], img_shape[1]
    ratio = W / H
    h = int(np.round(np.sqrt(num_patches / ratio)))
    w = int(np.round(np.sqrt(num_patches * ratio)))
    if hasattr(patches, 'reshape'):
        return patches.reshape(1, h, w, -1)
    return np.reshape(patches, (1, h, w, -1))


def visualize_features(last_hidden_state, img_shape):
    """Visualize DINOv2 patch features using PCA.

    Args:
        last_hidden_state: torch.Tensor or numpy.ndarray of shape (1, num_patches+1, 768).
        img_shape: tuple (H, W) of the input image shape, used to resize the visualization.
    """
    from sklearn.decomposition import PCA

    # Reshape to spatial grid: (1, num_patches+1, 768) -> (1, h, w, 768)
    grid = reshape_patches(last_hidden_state, img_shape)

    # Convert to numpy
    if hasattr(grid, 'cpu'):
        grid = grid.cpu().detach().numpy()
    grid_h, grid_w = grid.shape[1], grid.shape[2]
    patch_features = grid.reshape(-1, grid.shape[-1])  # (h*w, 768)

    H, W = img_shape[0], img_shape[1]

    # PCA to reduce 768-d features to 3 components for RGB visualization
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(patch_features)  # (h*w, 3)

    # Normalize each component to [0, 255]
    for i in range(3):
        min_val = pca_features[:, i].min()
        max_val = pca_features[:, i].max()
        if max_val - min_val > 1e-8:
            pca_features[:, i] = (pca_features[:, i] - min_val) / (max_val - min_val)
        else:
            pca_features[:, i] = 0.0

    pca_features = (pca_features * 255).astype(np.uint8)

    # Reshape to spatial grid and resize to input image dimensions
    feature_image = pca_features.reshape(grid_h, grid_w, 3)
    feature_image = cv2.resize(feature_image, (W, H), interpolation=cv2.INTER_NEAREST)

    # Convert RGB to BGR for cv2.imwrite
    feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2BGR)

    return feature_image