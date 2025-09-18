import random 
import numpy as np
import torch
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_multiple_axes(dfs: List[pd.DataFrame], columns_to_plot: Dict[int, List[str]]):
    """
    Plots specified columns from a list of DataFrames on separate y-axes.

    Args:
        dfs (List[pd.DataFrame]): A list of pandas DataFrames to plot. All DataFrames
                                  are expected to have a common index (e.g., datetime).
        columns_to_plot (Dict[int, Dict[str, list|str]]): A dictionary mapping an integer
                                                 (representing an axis index) to a dict of
                                                 column names + plot info to be plotted on that axis.
    """
    if not dfs:
        raise ValueError("The list of DataFrames cannot be empty.")

    # Create the primary plot
    n_axes = len(columns_to_plot)
    fig, axs = plt.subplots(n_axes, figsize=(12, 8), sharex=True)

    # Iterate through the dictionary to create subsequent axes
    for axis_index, plot_metadata in columns_to_plot.items():
        current_ax = axs[axis_index]

        # Plot all specified columns for the current axis
        for col in plot_metadata['tags']:
            for df in dfs:
                if col in df.columns:
                    current_ax.plot(df.index, df[col], label=col)

        # Set the label for the current y-axis
        current_ax.set_ylabel(plot_metadata['ylabel'])

    # Set common x-axis label and title
    for ax in axs:
        ax.legend(loc='best')
    axs[-1].set_xlabel('Training steps')

    plt.tight_layout()
    plt.show()
