import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from patchify import patchify
from PIL import ImageOps


def extract_patches_and_resize(img_np, patch_size=128, out_size=256, overlap=0):
    """
    Extract patches of size patch_size x patch_size, then resize to out_size x out_size.
    Optionally, use overlap (in pixels).
    Returns a list of (patch_resized, row_idx, col_idx).
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
    Batch processes all images in input_folder, extracting patches and resizing.
    Each output is saved as <original_basename>_patch_{row}_{col}.png in output_folder.
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


