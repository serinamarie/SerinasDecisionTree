# Serina's Decision Tree

Implementation of a decision tree algorithm with only base Python and numpy. See [blogpost](https://medium.com/@serinagrill/what-is-a-decision-tree-classifier-f4bdf4be8d8b) for more info!

## Getting Started

Package Installation

```pip install SerinasDecisionTree==0.3```


```from SerinasDecisionTree.decisiontree import DecisionTreeClassifier
dtc = DecisionTreeClassifier() # default min_samples=2, max_depth=5
dtc.fit(X_train)
dtc.predict(X_val)
```

Hyperparameter Tuning

```min_samples, max_depth```

Additional Parameter

```Append: default None, else will append new column of predictions to existing dataframe.```


### More about decision trees

- [Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
