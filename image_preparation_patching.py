
"""
patching.py

Patch extraction utility for microscopy and other image data.

Features:
- Loads images and auto-handles EXIF rotation.
- Crops images to be patch-friendly.
- Extracts overlapping or non-overlapping patches.
- Saves patches in a batch, with optional grayscale conversion.

Requirements:
- numpy
- pillow (PIL)
- patchify
"""

import os
import numpy as np
from PIL import Image, ImageOps
from patchify import patchify

PATCH_SIZE = 256
STRIDE = 128

def load_image(path):
    """
    Load an image from file as a NumPy array.

    Automatically converts to grayscale if the image is not already 'L' mode.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    img_np : np.ndarray
        2D grayscale image as a NumPy array.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != 'L':
        img = img.convert('L')
    return np.array(img)


def create_patches(
    image,
    patch_size=256,
    stride=128,
    save_dir=None,
    base_name="image"
):
    """
    Extract overlapping patches from a 2D image, optionally save each patch as PNG.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image.
    patch_size : int, optional
        Size of square patches (default: 256).
    stride : int, optional
        Stride or step size (default: 128).
    save_dir : str, optional
        If provided, each patch will be saved as a PNG in this directory.
    base_name : str, optional
        Used as prefix for saved patch files.

    Returns
    -------
    patches : np.ndarray
        Array of shape (n_patches, patch_size, patch_size).
    """
    patches = patchify(image, (patch_size, patch_size), step=stride).reshape(-1, patch_size, patch_size)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for i, patch in enumerate(patches):
            out_name = f"{base_name}_patch_{i:03d}.png"
            Image.fromarray(patch.astype(np.uint8)).save(os.path.join(save_dir, out_name))
        print(f"Saved {len(patches)} patches to {save_dir}")
    return patches


def crop_to_patchable(image, patch_size=PATCH_SIZE, stride=STRIDE):
    """
    Crop image so that its dimensions are compatible with patchify.

    Ensures that patches fit exactly and avoids partial patches on the borders.

    Parameters
    ----------
    image : np.ndarray
        2D array (grayscale image).
    patch_size : int, optional
        Size of square patches (default: PATCH_SIZE).
    stride : int, optional
        Stride or step size (default: STRIDE).

    Returns
    -------
    cropped : np.ndarray
        Cropped image, patchify-compatible.
    """
    h, w = image.shape
    new_h = ((h - patch_size) // stride) * stride + patch_size
    new_w = ((w - patch_size) // stride) * stride + patch_size
    return image[:new_h, :new_w]


def patch_and_save_unlabeled(source_dir, output_dir, patch_size=PATCH_SIZE, stride=STRIDE):
    """
    Batch-process all images in a directory, extracting patches and saving them as individual PNG files.

    Patches are named <original_filename>_patch_<index>.png.
    Handles EXIF rotation and grayscale conversion automatically.

    Parameters
    ----------
    source_dir : str
        Path to the directory containing input images.
    output_dir : str
        Path to the directory where patches will be saved.
    patch_size : int, optional
        Size of square patches (default: PATCH_SIZE).
    stride : int, optional
        Stride or step size between patch starts (default: STRIDE).

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(source_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            continue

        img_path = os.path.join(source_dir, fname)
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)

        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        img_np = np.array(img)
        print(f"📷 {fname}: original shape {img_np.shape}")

        # Crop to be compatible with patchify
        img_np = crop_to_patchable(img_np, patch_size, stride)
        print(f"{fname}: cropped shape {img_np.shape}")

        patches = create_patches(img_np, patch_size, stride)

        base = os.path.splitext(fname)[0]
        for i, patch in enumerate(patches):
            Image.fromarray(patch.astype(np.uint8)).save(
                os.path.join(output_dir, f"{base}_patch_{i:03d}.png")
            )


