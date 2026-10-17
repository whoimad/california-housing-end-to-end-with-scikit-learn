"""
California Housing, End to End with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_housing
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

HOUSING_URL = "https://github.com/ageron/data/raw/main/housing.tgz"


def load_housing() -> pd.DataFrame:
    #Download (if needed) and load the California housing dataset.
    tarball_path = Path(tempfile.gettempdir()) / "housing.tgz"

    if not tarball_path.is_file():
        # Download to a temporary name first so an interrupted download
        # never leaves a truncated housing.tgz that later runs would trust.
        partial_path = tarball_path.with_suffix(".tgz.part")
        urllib.request.urlretrieve(HOUSING_URL, partial_path)
        os.replace(partial_path, tarball_path)

    with tarfile.open(tarball_path) as tar:
        member = tar.extractfile("housing/housing.csv")
        if member is None:
            raise FileNotFoundError("housing/housing.csv not found in tarball")
        with member:
            return pd.read_csv(member)

# Step 2 - income_categories
def income_categories(df):
    # pd.cut median_income with edges [0, 1.5, 3, 4.5, 6, inf] and labels 1..5; return an int Series.
    edges = [0, 1.5, 3.0, 4.5, 6.0, np.inf]
    labels = [1,2,3,4,5]
    df_cut = pd.cut(df['median_income'], bins = edges, labels= labels)
    return df_cut.astype(int)

# Step 3 - stratified_split
from sklearn.model_selection import train_test_split

def stratified_split(df, test_size=0.2, random_state=42):
    # train_test_split stratified on income_categories(df); return (train_set, test_set).
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state, stratify=income_categories(df))
    return (train_set, test_set)

# Step 4 - explore_correlations
def explore_correlations(df):
    # Pearson correlation of every numeric column with median_house_value, sorted descending, target excluded.
    corr_matrix = df.corr(numeric_only=True)
    correlations = corr_matrix["median_house_value"].drop("median_house_value")
    return correlations.sort_values(ascending=False)

# Step 5 - add_ratio_features
def add_ratio_features(df):
    #Return a copy with rooms_per_house, bedrooms_ratio and people_per_house columns added.
    df_copy = df.copy()
    df_copy['rooms_per_house'] = df_copy['total_rooms'] / df_copy['households']
    df_copy['bedrooms_ratio'] = df_copy['total_bedrooms'] / df_copy['total_rooms']
    df_copy['people_per_house'] = df_copy['population'] / df_copy['households']
    return df_copy

# Step 6 - split_features_labels
def split_features_labels(df):
    # Return (X without median_house_value, y = median_house_value Series).
    X = df.drop(columns=['median_house_value'])
    y = df['median_house_value']
    return X , y

# Step 7 - ClusterSimilarity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters= n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            self.n_clusters, n_init=10, random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        # rbf_kernel similarity of each row of X to the cluster centers
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# Step 8 - numeric_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def numeric_pipeline():
    Pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    return Pipeline

# Step 9 - categorical_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def categorical_pipeline():
    Pipeline = make_pipeline(SimpleImputer(strategy="most_frequent", missing_values=pd.NA), OneHotEncoder(handle_unknown="ignore"))
    return Pipeline

# Step 10 - build_preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

def build_preprocessing(n_clusters=10, gamma=1.0, random_state=42):
    # ColumnTransformer with 'log', 'geo', 'cat' transformers and remainder=numeric_pipeline().
    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler())

    return ColumnTransformer(
        [(
            "log", 
            log_pipeline, 
            ["total_bedrooms", "total_rooms", "population", "households", "median_income"]
        ),
        (
            "geo",
            ClusterSimilarity(n_clusters, gamma, random_state),
            ["latitude", "longitude"]
        ),
        ("cat", categorical_pipeline(), ["ocean_proximity"])

        ],
        remainder=numeric_pipeline(),
    )

# Step 11 - rmse
import numpy as np
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

# Step 12 - dummy_baseline_rmse
from sklearn.dummy import DummyRegressor

def dummy_baseline_rmse(X, y):
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X, y)
    return rmse(y, dummy.predict(X))

# Step 13 - cross_val_rmse
import numpy as np
from sklearn.model_selection import cross_val_score


def cross_val_rmse(model, X, y, cv=3):
    scores = -cross_val_score(
        model, X, y, scoring="neg_root_mean_squared_error", cv=cv
    )
    return {
        "scores": [float(s) for s in scores],
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }

# Step 14 - linear_model
from sklearn.linear_model import LinearRegression

def linear_model(preprocessing):
    Pipeline = make_pipeline(preprocessing, LinearRegression())
    return Pipeline

# Step 15 - forest_model
from sklearn.ensemble import RandomForestRegressor

def forest_model(preprocessing, n_estimators=50, random_state=42):
    Pipeline = make_pipeline(preprocessing, RandomForestRegressor(n_estimators=n_estimators, random_state=random_state))
    return Pipeline

# Step 16 - random_search
from sklearn.model_selection import RandomizedSearchCV

def random_search(pipeline, X, y, n_iter=5, cv=3, random_state=42):
    param_distribs = {
        "columntransformer__geo__n_clusters": range(3, 11),
        "randomforestregressor__max_features": range(2, 9),
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distribs,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
    )
    search.fit(X, y)
    return search

# Step 17 - test_rmse
def test_rmse(model, test_set):
    X_test, y_test = split_features_labels(add_ratio_features(test_set))
    predictions = model.predict(X_test)
    return rmse(y_test, predictions)

# Step 18 - bootstrap_rmse_ci
def bootstrap_rmse_ci(y_true, y_pred, n_boot=200, alpha=0.05, random_state=42):
    # Resample pairs with replacement, rescore, take the 2.5th-97.5th percentiles = 95% CI on RMSE
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(random_state)

    boot_rmses = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_rmses.append(rmse(y_true[idx], y_pred[idx]))

    low = np.percentile(boot_rmses, 100 * alpha / 2)
    high = np.percentile(boot_rmses, 100 * (1 - alpha / 2))
    return float(low), float(high)

# Step 19 - feature_importances
def feature_importances(search, k=5):
    best = search.best_estimator_
    importances = best[-1].feature_importances_
    names = best[0].get_feature_names_out()
    ranked = sorted(zip(importances, names), key=lambda pair: pair[0], reverse=True)
    return [(float(round(imp, 3)), str(name)) for imp, name in ranked[:k]]

# Step 20 - worst_errors (not yet solved)
# TODO: implement

# Step 21 - save_and_reload (not yet solved)
# TODO: implement

# Step 22 - predict_new (not yet solved)
# TODO: implement

