import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def extract_patches_and_resize(img_np, patch_size=128, out_size=256, overlap=0):
    """
    Extract overlapping patches from a 2D grayscale image and resize them.

    Parameters
    ----------
    img_np : np.ndarray
        Input 2D array (grayscale image).
    patch_size : int, optional
        Size of square patch to extract (default: 128).
    out_size : int, optional
        Size to resize each patch to (default: 256).
    overlap : int, optional
        Overlap between adjacent patches in pixels (default: 0).

    Returns
    -------
    patches : list of tuples
        List of (patch_resized, row_idx, col_idx), where:
            patch_resized : np.ndarray
                The resized patch as a 2D array (out_size x out_size).
            row_idx : int
                Row index (top left of the patch in original image).
            col_idx : int
                Column index (top left of the patch in original image).

    Notes
    -----
    - Uses PIL's bicubic interpolation for resizing.
    - Useful for creating patch datasets for ML or downstream analysis.
    """
    h, w = img_np.shape
    step = patch_size - overlap
    patches = []
    for i in range(0, h - patch_size + 1, step):
        for j in range(0, w - patch_size + 1, step):
            patch = img_np[i:i+patch_size, j:j+patch_size]
            patch_resized = np.array(Image.fromarray(patch).resize((out_size, out_size), resample=Image.BICUBIC))
            patches.append((patch_resized, i, j))
    return patches

def batch_patchify_and_resize(input_folder, output_folder, patch_size=128, out_size=256, overlap=0):
    """
    Batch-process all images in a folder, extracting and resizing patches.

    For each image:
    - Converts to grayscale (if not already).
    - Extracts patches of size patch_size x patch_size, with specified overlap.
    - Resizes each patch to out_size x out_size.
    - Saves each patch as a PNG: <original_basename>_patch_{row}_{col}.png.

    Parameters
    ----------
    input_folder : str
        Path to the input folder containing images (.png, .jpg, .jpeg, .tif, .tiff).
    output_folder : str
        Path to the output folder where patches will be saved.
    patch_size : int, optional
        Size of the square patch to extract from each image (default: 128).
    out_size : int, optional
        Size to resize each patch to before saving (default: 256).
    overlap : int, optional
        Overlap between patches in pixels (default: 0).

    Returns
    -------
    None

    Example
    -------
    >>> batch_patchify_and_resize(
    ...     "input_images/",
    ...     "output_patches/",
    ...     patch_size=128,
    ...     out_size=256,
    ...     overlap=32
    ... )
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    for fname in tqdm(image_files, desc="Processing images"):
        in_path = os.path.join(input_folder, fname)
        base = os.path.splitext(fname)[0]
        img = Image.open(in_path).convert("L")
        img_np = np.array(img)
        patches = extract_patches_and_resize(img_np, patch_size, out_size, overlap)
        for patch_resized, row, col in patches:
            out_name = f"{base}_patch_{row}_{col}.png"
            out_path = os.path.join(output_folder, out_name)
            Image.fromarray(patch_resized).save(out_path)


# Example usage:
input_folder = "root directory of images"
output_folder = "destination folder"

batch_patchify_and_resize(input_folder, output_folder, patch_size=128, out_size=256, overlap=0)
