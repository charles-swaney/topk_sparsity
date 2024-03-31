import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from read_logs import read_logs_to_df
from typing import List


def generate_plot_from_logs(logs_dir, type: str, models: List[str]):
    """
    Generate plots for either training or validation for a specified subset of models.

    Arguments:
        - logs_dir: the path to the directory storing training/val logs
        - type: either Train or Validation
        - models: a subset of models for which to generate the plots

    Outputs:
        - a plot of training/val data
    """
    try:
        df = read_logs_to_df(logs_dir)
        sns.set_theme(style='darkgrid')
        plt.figure(figsize=(12,8))
        sns.lineplot(data=df, x='Epoch', y='Value', hue='Model', style='Type', markers=True, dashes=False)
        plt.title('Model Performance Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend(title='Model and Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("File not found, check path.")
    except pd.errors.EmptyDataError:
        print("One or more entries is empty.")
    except Exception as e:
        print(f"An error occurred: {e}.") 
