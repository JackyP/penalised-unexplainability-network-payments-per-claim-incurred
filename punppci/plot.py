import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_triangle(data, mask_bottom=True):
    """ Plots a claims triangle.
    data:
        Pandas DataFrame, indexed by date

    mask_bottom:
        Hide bottom half of triangle. Assumes origin and development periods
        are the same. e.g. both months
    """
    sns.set(style="white")

    # Generate a mask for the triangle
    if mask_bottom:
        mask = np.ones_like(data, dtype=np.bool)

        mask[np.tril_indices_from(mask)] = False

        mask = np.flipud(mask)
    else:
        mask = None

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 10))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio vmax=.3,
    sns.heatmap(
        data,
        mask=mask,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        annot_kws={"size": 9},
        fmt=".0f",
        cbar_kws={"shrink": 0.5},
    )

    ax.set_yticklabels(labels=data.index.strftime("%Y-%m-%d"))

    return f, ax
