"""
Script to train Random Forest
"""
import numpy as np
import pandas as pd
from utils import entropy, gini, misclassification_error, get_train_val_split




def main():
    pth = "../data/agaricus-lepiota-training.csv"
    df = pd.read_csv(pth)
    train_df, val_df = get_train_val_split(df, val_r=0.2, seed=42)
    feat_rs = np.arange(0.1, 0.8, 0.1)
    metrics = [entropy, gini, misclassification_error]
    alpha = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99]
    bag_r = np.arange(0.1, 0.8, 0.1)
    tree_count = np.arange(10, 110, 10)
    seed_for_bag_seed_generator = 42  # random
    seed_for_feat_bag_seed_generator = 42  # random
    acc_map = {}



if __name__ == "__main__":
    main()
