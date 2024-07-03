import pandas as pd, numpy as np, re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor


# from utils.mlp import MLP


def parse_data(filename):
    with open(filename, "rb") as f:
        infoline = f.readline()
        infoline = re.sub(r"^b'", "", str(infoline))
        n_features = int(re.sub(r"^\d+\s(\d+)\s\d+.*$", r"\1", infoline))
        features, labels = load_svmlight_file(f, n_features=n_features, multilabel=True)
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    features = np.array(features.todense())
    features = np.ascontiguousarray(features)
    return features, labels


# def create_reward_func(reward_func, feature_dim, reward_dim):
    
#     if reward_func == "logistic_regression":
        
#         return LogisticRegression()
    
#     elif reward_func == "MLP":
        
#         return MLP(feature_dim, reward_dim)
    


    
    