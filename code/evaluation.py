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

    print("tree_count avg acc ", results_df.groupby(["tree_count"])["accuracy"].mean())
    print("tree_count min acc ", results_df.groupby(["tree_count"])["accuracy"].min())
    print("tree_count max acc ", results_df.groupby(["tree_count"])["accuracy"].max())

    print("metric fn avg acc ", results_df.groupby(["metric_fn"])["accuracy"].mean())
    print("metric fn min acc ", results_df.groupby(["metric_fn"])["accuracy"].min())
    print("metric fn max acc ", results_df.groupby(["metric_fn"])["accuracy"].max())

    print("bag_r avg acc ", results_df.groupby(["bag_r"])["accuracy"].mean())
    print("bag_r min acc ", results_df.groupby(["bag_r"])["accuracy"].min())
    print("bag_r max acc ", results_df.groupby(["bag_r"])["accuracy"].max())

    print("feature_r avg acc ", results_df.groupby(["feature_r"])["accuracy"].mean())
    print("feature_r min acc ", results_df.groupby(["feature_r"])["accuracy"].min())
    print("feature_r max acc ", results_df.groupby(["feature_r"])["accuracy"].max())

    print("alpha avg acc ", results_df.groupby(["alpha"])["accuracy"].mean())
    print("alpha min acc ", results_df.groupby(["alpha"])["accuracy"].min())
    print("alpha max acc ", results_df.groupby(["alpha"])["accuracy"].max())

    print("top_feature avg acc ", results_df.groupby(["top_feature"])["accuracy"].mean())
    print("top_feature min acc ", results_df.groupby(["top_feature"])["accuracy"].min())
    print("top_feature max acc ", results_df.groupby(["top_feature"])["accuracy"].max())

    print("max_depth avg acc ", results_df.groupby(["max_depth"])["accuracy"].mean())
    print("max_depth min acc ", results_df.groupby(["max_depth"])["accuracy"].min())
    print("max_depth max acc ", results_df.groupby(["max_depth"])["accuracy"].max())

if __name__ == "__main__":
    results_pth = "../data/grid_searchmike_4hr_val_r_99.csv"
    main(results_pth)
