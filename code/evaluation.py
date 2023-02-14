import pandas as pd
import numpy as np

"""
@author Jack Ringer, Mike Adams
Date:2/13/2023
Description: Script to show which params gave the best (validation) accuracy
"""


def main(csv_path: str, acc_col: str = "accuracy"):
    results_df = pd.read_csv(csv_path)
    print("All results:")
    print(results_df)
    print("-" * 80)
    best_idx = np.argmax(results_df[acc_col])
    worst_idx = np.argmin(results_df[acc_col])
    print(set(results_df[acc_col]))
    print("Best params found were:", results_df.iloc[best_idx])
    print("-" * 80)
    print("Worst params found were:", results_df.iloc[worst_idx])


if __name__ == "__main__":
    results_pth = "../data/grid_searchmike.csv"
    main(results_pth)
