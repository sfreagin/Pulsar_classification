# Pulsar Classification

Hello,

This repository uses publicly-available statistical data on suspected pulsars to classify radio telescope observations as Noise or Pulsar. 
The dataset was downloaded from the [UCI repository](https://archive.ics.uci.edu/ml/index.php) at this address: https://archive.ics.uci.edu/ml/datasets/HTRU2

### Navigation
The dataset (including original readme.txt) can be found in the `/dataset` folder. 
I created full Python scripts in the `/py_scripts` folder and accompanying Jupyter notebooks in the `/notebooks` folder.

### Purpose of this repo
Each python script utilizes a different machine learning technique -- LogisticRegression, NaiveBayes, KMeans, DecisionTree, RandomForest, neural net -- 
to predict whether a given observation should be recorded as **0** (noise) or **1** (pulsar)
