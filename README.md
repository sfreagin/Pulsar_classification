# Pulsar Classification

Hello,

This repository uses publicly-available statistical data on suspected pulsars to classify radio telescope observations as Noise or Pulsar. 
The dataset was downloaded from the [UCI repository](https://archive.ics.uci.edu/ml/index.php) at this address: https://archive.ics.uci.edu/ml/datasets/HTRU2

### Navigation
The dataset (including original readme.txt) can be found in the `/dataset` folder. 
I created full Python scripts in the `/py_scripts` folder and accompanying Jupyter notebooks in the `/notebooks` folder.

### Purpose of this repo
Each python script utilizes a different machine learning technique -- DecisionTree, KNeighbors, LogisticRegression, NaiveBayes, RandomForest, neural net -- 
to classify a given observation as **0** (noise) or **1** (pulsar)

## About the HTRU2 dataset
The data itself consists only of the Big 4 [statistical moments](https://en.wikipedia.org/wiki/Moment_(mathematics)):
- Mean
- Standard deviation
- Skewness
- Kurtosis
- DM-adjusted mean, DM-adjusted stdev, etc. Check the `/dataset/readme.txt` which references [this paper](http://www.scienceguyrob.com/wp-content/uploads/2016/12/WhyArePulsarsHardToFind_Lyon_2016.pdf) for more context

The dataset is unbalanced -- something like 90% of observations are radio noise and only 10% are pulsars. I plan to learn more about the parameters of each ML algorithm and make future adjustments to minimize false negatives here.
