import numpy as np
import matplotlib.pyplot as plt

def overlay_peaks_on_image(img_np, peaks, color='r', marker='o', markersize=4, ax=None, show=True, title=None):
    """
    Overlay detected peaks on a grayscale image.

    Args:
        img_np (np.ndarray): 2D grayscale image.
        peaks (np.ndarray): Nx2 array of (row, col) peak positions.
        color (str): Marker color.
        marker (str): Marker style.
        markersize (int): Marker size.
        ax (matplotlib.axes): Optionally plot on an existing axis.
        show (bool): Call plt.show() at the end.
        title (str): Optional title.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img_np, cmap='gray')
    if peaks is not None and len(peaks) > 0:
        ax.plot(peaks[:,1], peaks[:,0], marker=marker, linestyle='', color=color, markersize=markersize)
    if title:
        ax.set_title(title)
    ax.axis('off')
    if show:
        plt.show()


def show_displacement_map(disp_map, title=None):
    """
    Display a signed displacement map (continuous values) as a grayscale image.

    Parameters
    ----------
    disp_map : np.ndarray
        Displacement map scaled to [0, 255], dtype=uint8, 128 is neutral.
    title : str, optional
        Plot title.
    """
    plt.imshow(disp_map, cmap='gray', vmin=0, vmax=255)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def plot_grid_of_images(image_list, titles=None, ncols=3, figsize=(12,12)):
    """
    Display a grid of images (e.g. parameter sweeps).

    Args:
        image_list (list): List of 2D arrays to plot.
        titles (list): Optional titles.
        ncols (int): Number of columns.
        figsize (tuple): Figure size.
    """
    n = len(image_list)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    for i, img in enumerate(image_list):
        axes[i].imshow(img, cmap='gray')
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
        axes[i].axis('off')
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()


