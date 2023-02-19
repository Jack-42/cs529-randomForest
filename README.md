# RandomForest: CS529 Project 1

#### By Jack Ringer and Michael Adams

In this project we implement Random Forests to tackle the problem of mushroom
classification. See project report for more details.

## Project Guide

All code files used can be found in the code/ directory.

An overview of what is implemented in each of these files is given as follows:

* TreeNode.py: single decision tree node
* DecTree.py: decision tree
* Forest.py: random forest
* utils.py: various utility functions, including information metrics (e.g.,
  entropy)
* training.py: script used to train random forests using a grid-search of
  hyperparameters
* Results.py: helper class used to store configurations and validation accuracy
  measurements for trained forests
* evaluation.py: helper used to evaluate results from training.py
* plottopfeatuers.py: script used to plot the figure found in the appendix of
  the project paper

## Setup

All necessary packages to run the code in this project can be installed with:

pip install -r requirements.txt

