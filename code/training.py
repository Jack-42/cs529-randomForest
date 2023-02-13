"""
Script to train Random Forest
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import entropy, gini, misclassification_error, get_train_val_split, get_accuracy
from Forest import RandomForest
from Results import Results



def main():
    pth = "../data/agaricus-lepiota-training.csv"
    df = pd.read_csv(pth)
    split_seeds = np.arange(1, 4, 1)
    feat_rs = np.arange(0.1, 0.8, 0.1)
    metrics = {"entropy": entropy, "gini": gini, "misclassification_error": misclassification_error}
    alphas = [0.05]
    bag_rs = np.arange(0.1, 0.8, 0.1)
    tree_counts = np.arange(10, 20, 10)
    seeds_for_bag_seed_generator = np.arange(1, 4, 1)
    seeds_for_feat_bag_seed_generator = np.arange(1, 4, 1)
    
    """
    Parameter order for results: "feature_r", "metric_fn", "alpha", "bag_r", "tree_count",
                     "split_seed", "max_depth", "avg_depth", "accuracy",
                     "seed_for_bag_seed_generator",
                     'seed_for_feat_bag_seed_generator'
    """
    res = Results()

    pbar = tqdm(total=len(split_seeds) * len(seeds_for_bag_seed_generator) * len(seeds_for_feat_bag_seed_generator) * len(tree_counts) * len(bag_rs) * len(feat_rs) * len(alphas) * len(metrics))

    for split_seed in split_seeds:
        train_df, val_df = get_train_val_split(df, val_r=0.2, seed=split_seed)
        train_with = train_df.drop(columns="id")
        val_df_no_class = val_df.drop(columns="class")
        val_df_set_index = val_df.set_index('id')
        for bag_seed in seeds_for_bag_seed_generator:
            for feat_bag_seed in seeds_for_feat_bag_seed_generator:
                for tree_count in tree_counts:
                    for bag_r in bag_rs:
                        for feat_r in feat_rs:
                            for alpha in alphas:
                                for metric in metrics:
                                    forest = RandomForest(feat_r, metrics[metric], alpha, bag_r, tree_count, bag_seed, feat_bag_seed)
                                    forest.train(train_with)
                                    classifications = forest.classify(val_df_no_class)
                                    acc = get_accuracy(classifications["class"], val_df_set_index["class"])
                                    avg_depth = forest.avg_tree_depth()
                                    max_depth = forest.max_tree_depth()
                                    res.add_entry([feat_r, metric, alpha, bag_r, tree_count, split_seed, max_depth, avg_depth, acc, bag_seed, feat_bag_seed])
                                    pbar.update(1)

    pbar.close()

    res.results_df.to_csv("../data/grid_searchmike.csv")

if __name__ == "__main__":
    main()
