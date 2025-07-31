import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import os
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA         
from scipy.interpolate import griddata   
import itertools
from displacement import generate_signed_displacement_map, generate_unsigned_displacement_map




def fit_lattice(peaks, shape):
    """
    Fit a 2D Bravais lattice to detected atomic peaks using PCA and least squares.

    Parameters
    ----------
    peaks : np.ndarray
        Nx2 array of (row, col) positions of detected peaks.
    shape : tuple
        Shape of the original image (for reference).

    Returns
    -------
    i : np.ndarray or None
        Integer indices along first lattice vector.
    j : np.ndarray or None
        Integer indices along second lattice vector.
    lattice_func : callable or None
        Function mapping (i, j) to reference positions. None if failed.
    """
    from sklearn.decomposition import PCA
    if len(peaks) < 5:
        return None, None, None
    centered = peaks - peaks.mean(axis=0)
    pca = PCA(n_components=2)
    pca.fit(centered)
    lattice_vectors = pca.components_
    proj = centered @ lattice_vectors.T
    spacing = np.median(np.diff(np.sort(proj[:,0])))
    i = np.round(proj[:,0] / spacing).astype(int)
    j = np.round(proj[:,1] / spacing).astype(int)
    A = np.c_[np.ones_like(i), i, j]
    x_res, _, _, _ = np.linalg.lstsq(A, peaks[:,1], rcond=None)
    y_res, _, _, _ = np.linalg.lstsq(A, peaks[:,0], rcond=None)
    def lattice_func(ii, jj):
        x = x_res[0] + x_res[1]*ii + x_res[2]*jj
        y = y_res[0] + y_res[1]*ii + y_res[2]*jj
        return np.column_stack([y, x])
    return i, j, lattice_func


def compute_displacements(peaks, i, j, lattice_func):
    """
    Compute displacement vectors for each detected peak relative to ideal lattice.

    Parameters
    ----------
    peaks : np.ndarray
        Nx2 array of refined peak positions.
    i : np.ndarray
        Array of lattice indices along vector 1.
    j : np.ndarray
        Array of lattice indices along vector 2.
    lattice_func : callable
        Function mapping (i, j) to reference positions.

    Returns
    -------
    displacements : np.ndarray
        Nx2 array of (row_disp, col_disp) vectors.
    """
    ref_sites = lattice_func(i, j)
    return peaks - ref_sites

def find_peaks(
    img_np,
    sigma=1,
    min_distance=5,
    threshold_abs=30
):
    """
    Find peaks in a 2D grayscale image using Gaussian blur and skimage's peak_local_max.

    Parameters:
        img_np (np.ndarray): Grayscale image.
        sigma (float): Gaussian blur sigma before peak finding.
        min_distance (int): Minimum number of pixels separating peaks.
        threshold_abs (float): Minimum intensity of peaks.

    Returns:
        np.ndarray: Array of (row, col) peak positions.
    """
    from skimage.feature import peak_local_max
    from scipy.ndimage import gaussian_filter

    blurred = gaussian_filter(img_np, sigma=sigma)
    peaks = peak_local_max(
        blurred,
        min_distance=min_distance,
        threshold_abs=threshold_abs
    )
    return peaks



def displacement_field_to_image(peaks, displacements, image_shape, component='magnitude', method='cubic'):
    """
    Interpolate atomic displacements onto a full-resolution image grid.

    Parameters
    ----------
    peaks : np.ndarray
        Nx2 array of atomic positions (row, col).
    displacements : np.ndarray
        Nx2 array of displacement vectors (row, col).
    image_shape : tuple
        Output image shape (rows, cols).
    component : str, optional
        'magnitude', 'row', or 'col' (default is 'magnitude').
    method : str, optional
        Interpolation type: 'cubic', 'linear', or 'nearest' (default is 'cubic').

    Returns
    -------
    disp_img : np.ndarray
        Interpolated displacement map, scaled to [0, 255] uint8.
    """
    grid_y, grid_x = np.mgrid[0:image_shape[0], 0:image_shape[1]]
    if component == 'magnitude':
        values = np.linalg.norm(displacements, axis=1)
    elif component == 'row':
        values = displacements[:, 0]
    elif component == 'col':
        values = displacements[:, 1]
    else:
        raise ValueError("component must be 'magnitude', 'row', or 'col'")
    disp_img = griddata(peaks, values, (grid_y, grid_x), method=method, fill_value=0)
    disp_img = disp_img - np.nanmin(disp_img)
    if np.nanmax(disp_img) > 0:
        disp_img = disp_img / np.nanmax(disp_img) * 255
    disp_img = np.nan_to_num(disp_img, nan=0.0)
    return disp_img.astype(np.uint8)


def parameter_grid(
    image_path,
    sigmas=[1, 2],
    min_distances=[5, 7, 9],
    thresholds=[30, 40, 50],
    blur_ksize=(3,3),
    sobel_ksize=3,
    max_plots=4,
    use_unsigned_disp_map=True,
    component='magnitude'
):
    """
    Visualize grid search for peak finding and displacement mapping parameters.

    For each parameter set, displays a figure with three columns:
        1. Displacement map only (gradient-based, signed). This map is directly calculated from the Sobel gradient of the image.
        2. Original image with detected peaks overlaid (for peak QC).
        3. Detected peaks as red dots on a blank (white) background.

    Parameters
    ----------
    image_path : str
        Path to the image to analyze.
    sigmas : list of int, optional
        List of Gaussian blur sigmas for peak finding (default [1,2]).
    min_distances : list of int, optional
        List of min_distance for peak finding (default [5,7,9]).
    thresholds : list of float, optional
        List of threshold_abs for peak finding (default [30,40,50]).
    blur_ksize : tuple of int, optional
        Kernel size for Gaussian blur in gradient map (default (3,3)).
    sobel_ksize : int, optional
        Kernel size for Sobel operator in gradient map (default 3).
    max_plots : int, optional
        Maximum number of parameter sets to plot (default 4).
    component : str, optional
        Component of displacement to visualize: 'magnitude', 'row', or 'col' (default 'magnitude').

    Returns
    -------
    None. (Displays a matplotlib figure grid.)
    """

    img = Image.open(image_path).convert("L")
    img_np = np.array(img)
    param_combos = list(itertools.product(sigmas, min_distances, thresholds))
    if len(param_combos) > max_plots:
        print(f"Too many parameter sets ({len(param_combos)}). Showing first {max_plots}.")
        param_combos = param_combos[:max_plots]
    nrows = len(param_combos)
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    for row, (sigma, min_distance, threshold_abs) in enumerate(param_combos):
        # 1. Peak finding
        blurred = gaussian_filter(img_np, sigma=sigma)
        peaks = peak_local_max(
            blurred,
            min_distance=min_distance,
            threshold_abs=threshold_abs
        )
        # 1st column: signed gradient-based displacement map
        disp_map = generate_signed_displacement_map(img_np, blur_ksize=blur_ksize, sobel_ksize=sobel_ksize)
        ax_disp = axes[row, 0]
        ax_disp.imshow(disp_map, cmap='gray')
        ax_disp.set_title(f"Disp map (signed)\nsigma={sigma}, md={min_distance}, thr={threshold_abs}")
        ax_disp.axis('off')

        # 2nd column: overlay peaks
        ax_overlay = axes[row, 1]
        ax_overlay.imshow(img_np, cmap='gray')
        if peaks is not None and len(peaks) > 0:
            ax_overlay.plot(peaks[:,1], peaks[:,0], 'r.', markersize=4, alpha=0.8)
            ax_overlay.set_title(f"Overlay | N={len(peaks)}")
        else:
            ax_overlay.set_title(f"Overlay | NO PEAKS")
        ax_overlay.axis('off')

        # 3rd column: just detected peaks as red dots on blank background
        ax_peaks = axes[row, 2]
        peak_img = np.ones_like(img_np) * 255  # or 0 for black
        ax_peaks.imshow(peak_img, cmap='gray')
        if peaks is not None and len(peaks) > 0:
            ax_peaks.plot(peaks[:,1], peaks[:,0], 'r.', markersize=4, alpha=0.8)
            ax_peaks.set_title(f"Peaks only | N={len(peaks)}")
        else:
            ax_peaks.set_title("Peaks only | NO PEAKS")
        ax_peaks.axis('off')

    plt.tight_layout()
    plt.show()


def batch_peak_find_and_displacement(
    input_folder,
    output_folder,
    sigma=1,
    min_distance=5,
    threshold_abs=30,
    blur_ksize=(3,3),
    sobel_ksize=3,
    save_overlay=True,
    save_peaks=True,
    save_disp=True,
    component='magnitude'
):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    for fname in image_files:
        img_path = os.path.join(input_folder, fname)
        base = os.path.splitext(fname)[0]
        img = Image.open(img_path).convert("L")
        img_np = np.array(img)

        # Peak finding
        blurred = gaussian_filter(img_np, sigma=sigma)
        peaks = peak_local_max(blurred, min_distance=min_distance, threshold_abs=threshold_abs)

        if save_peaks:
            np.save(os.path.join(output_folder, f"{base}_peaks.npy"), peaks)

        if save_overlay:
            overlay = np.stack([img_np]*3, axis=-1)
            for r, c in peaks:
                overlay[max(r-2,0):r+3, max(c-2,0):c+3, 0] = 255
                overlay[max(r-2,0):r+3, max(c-2,0):c+3, 1:] = 0
            Image.fromarray(overlay).save(os.path.join(output_folder, f"{base}_peaks_overlay.png"))

        # Displacement field workflow:
        # 1. Fit lattice
        i, j, lattice_func = fit_lattice(peaks, img_np.shape)
        if lattice_func is not None:
            # 2. Compute displacements
            displacements = compute_displacements(peaks, i, j, lattice_func)
            np.save(os.path.join(output_folder, f"{base}_displacements.npy"), displacements)
            # 3. Interpolate to full image grid
            disp_img = displacement_field_to_image(peaks, displacements, img_np.shape, component=component)
            if save_disp:
                Image.fromarray(disp_img).save(os.path.join(output_folder, f"{base}_dispfield_{component}.png"))
                np.save(os.path.join(output_folder, f"{base}_dispfield_{component}.npy"), disp_img)
        else:
            print(f"❌ Lattice fitting failed for {fname} (not enough peaks or ill-conditioned)")

    print(f"✅ Batch peak finding and displacement field workflow complete for {len(image_files)} images.")


