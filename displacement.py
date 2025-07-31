import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm    
import itertools        
import math              
import matplotlib.pyplot as plt 

def generate_signed_displacement_map(image_np, blur_ksize=(5, 5), sobel_ksize=3):
    """
    Generate a displacement map that marks atomic plane centers only (no doubling), 
    by identifying zero-crossings of the vertical Sobel gradient.

    This function finds where the sign of the gradient changes along the vertical direction,
    corresponding to the center of each atomic plane. The output is a binary image:
    atomic plane centers are marked as 255, background as 0.

    Parameters
    ----------
    image_np : np.ndarray
        2D grayscale input image.
    blur_ksize : tuple of int, optional
        Kernel size for Gaussian blur (default is (5,5)).
    sobel_ksize : int, optional
        Kernel size for Sobel operator (default is 3).

    Returns
    -------
    disp_map : np.ndarray
        Binary uint8 image, same shape as input. Atomic plane centers = 255, background = 0.
    """
    blurred = cv2.GaussianBlur(image_np, blur_ksize, 0)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, dx=0, dy=1, ksize=sobel_ksize)
    max_val = np.max(np.abs(sobel_y))
    if max_val == 0:
        return np.full_like(image_np, 128, dtype=np.uint8)
    norm = sobel_y / max_val
    norm_shifted = ((norm + 1) / 2) * 255
    return norm_shifted.astype(np.uint8)

def generate_unsigned_displacement_map(image_np, blur_ksize=(3, 3), sobel_ksize=3):
    """
    Compute an unsigned vertical gradient displacement map 
    similar to your 'expected' output (atomic-scale sharpness).
    
    Parameters
    ----------
    image_np : np.ndarray
        2D grayscale image.
    blur_ksize : tuple, optional
        Gaussian blur kernel size (default (3,3)).
    sobel_ksize : int, optional
        Sobel kernel size (default 3).
    
    Returns
    -------
    disp_map : np.ndarray
        2D array, uint8, scaled to [0, 255].
    """
    if blur_ksize[0] > 1:
        blurred = cv2.GaussianBlur(image_np, blur_ksize, 0)
    else:
        blurred = image_np
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, dx=0, dy=1, ksize=sobel_ksize)
    disp = np.abs(sobel_y)
    if disp.ptp() == 0:
        return np.zeros_like(image_np, dtype=np.uint8)
    norm = ((disp - disp.min()) / disp.ptp()) * 255
    return norm.astype(np.uint8)


def batch_generate_displacement_map_png(
    input_folder,
    output_folder,
    use_signed=True,
    disp_map_params=None
):
    """
    Batch-process microscopy images to generate and save displacement maps as PNG files.

    This function reads all supported image files in the input folder, computes a displacement
    map for each image (signed or unsigned), and saves the resulting map as a PNG to the output folder.

    Parameters:
        input_folder (str): Path to the folder containing input microscopy images
                            (supports .png, .jpg, .jpeg, .tif, .tiff).
        output_folder (str): Path to the folder where output PNG displacement maps will be saved.
        use_signed (bool): Whether to use the signed displacement map (default: True).
        disp_map_params (dict, optional): Additional parameters to pass to the displacement map
                                          function (e.g., {'blur_ksize': (7,7), 'sobel_ksize': 5}).

    Saves:
        For each input image, a PNG displacement map is saved in the output folder,
        named as '<original_basename>_disp.png'.

    Example:
        batch_generate_displacement_map_png(
            input_folder='input_dir',
            output_folder='output_dir',
            use_signed=False,
            disp_map_params={'blur_ksize': (7,7), 'sobel_ksize': 5}
        )
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    disp_map_params = disp_map_params or {}
    for fname in tqdm(image_files, desc="Processing images"):
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, os.path.splitext(fname)[0] + "_disp.png")
        img = Image.open(in_path).convert("L")
        img_np = np.array(img)
        if use_signed:
            disp_map = generate_signed_displacement_map(img_np, **disp_map_params)
        else:
            disp_map = generate_displacement_map(img_np, **disp_map_params)
        Image.fromarray(disp_map).save(out_path)




def tune_displacement_params(
    image_path,
    param_grid=None,
    use_signed=True,
    save_outputs=False,
    output_folder=None,
    show_plots=True
):
    """
    Try all combinations of given parameter options for the displacement map function
    on a single image and plot or save results.

    Parameters:
        image_path (str): Path to the image file.
        param_grid (dict): Dictionary of parameter lists, e.g.
            {
                'blur_ksize': [(5,5), (11,11)],
                'sobel_ksize': [3, 5, 7],
                'denoise_h': [None, 10],
                'use_equalize': [False, True]
            }
        use_signed (bool): Use signed or unsigned displacement map.
        save_outputs (bool): Save each output as PNG if True.
        output_folder (str): Where to save outputs. If None, images will not be saved.
        show_plots (bool): Whether to display each result as a plot.

    Example:
        param_grid = {
            'blur_ksize': [(5,5), (11,11)],
            'sobel_ksize': [3, 7],
            'denoise_h': [None, 10],
            'use_equalize': [False, True]
        }
        tune_displacement_params("img.png", param_grid, use_signed=True)
    """
    if param_grid is None:
            param_grid = {
                'blur_ksize': [(5,5), (9,9), (11,11)],
                'sobel_ksize': [3, 5, 7]
            }

    keys, values = zip(*param_grid.items())
    combos = list(itertools.product(*values))

    img = Image.open(image_path).convert("L")
    img_np = np.array(img)

    n_combos = len(combos)
    n_cols = 2
    n_rows = math.ceil(n_combos / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))
    axes = axes.flatten()  # Easier indexing

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        if use_signed:
            disp_map = generate_signed_displacement_map(img_np, **params)
        else:
            disp_map = generate_displacement_map(img_np, **params)
        desc = ', '.join(f"{k}={params[k]}" for k in keys)
        ax = axes[idx]
        ax.imshow(disp_map, cmap='gray')
        ax.set_title(desc, fontsize=10)
        ax.axis('off')

        # Optionally save
        if save_outputs and output_folder:
            import os
            os.makedirs(output_folder, exist_ok=True)
            param_str = '_'.join(f"{k}{str(params[k]).replace(',','x')}" for k in keys)
            out_name = f"{param_str}_disp.png"
            Image.fromarray(disp_map).save(os.path.join(output_folder, out_name))

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    if show_plots:
        plt.tight_layout()
        plt.show()
