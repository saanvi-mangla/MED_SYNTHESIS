import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
def create_overlay(background, overlay, alpha=0.5, cmap='hot'):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(background, cmap='gray', origin='lower')
    ax.imshow(overlay, cmap=cmap, alpha=alpha, origin='lower')
    ax.axis('off')
    return fig
def create_montage(slices, num_cols=4):
    num_slices = len(slices)
    num_rows = int(np.ceil(num_slices / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3, num_rows*3))
    axes = axes.flatten()
    for i, slice_img in enumerate(slices):
        axes[i].imshow(slice_img, cmap='gray', origin='lower')
        axes[i].axis('off')
        axes[i].set_title(f'Slice {i}')
    for i in range(num_slices, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    return fig
def window_level_ct(ct_array, window=400, level=40):
    min_val = level - window / 2
    max_val = level + window / 2
    windowed = np.clip(ct_array, min_val, max_val)
    windowed = (windowed - min_val) / (max_val - min_val)
    return windowed
