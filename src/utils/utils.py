import pandas as pd, numpy as np, re
from copy import deepcopy
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from contextualbandits.online import AdaptiveGreedy


from utils.mlp import MLP


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


def create_reward_func(reward_func, feature_dim, reward_dim):
    
    if reward_func == "logistic_regression":
        
        base_algorithm = LogisticRegression(solver='lbfgs', warm_start=True)
        beta_prior = ((3./reward_dim, 4), 2)
        return AdaptiveGreedy(deepcopy(base_algorithm), nchoices=reward_dim,
                                     decay_type='threshold',
                                     beta_prior = beta_prior, random_state = 6666)
    
    elif reward_func == "MLP":
        
        return MLP(feature_dim, reward_dim)
    


    
    