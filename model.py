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

# Step 5 - add_ratio_features (not yet solved)
# TODO: implement

# Step 6 - split_features_labels (not yet solved)
# TODO: implement

# Step 7 - ClusterSimilarity (not yet solved)
# TODO: implement

# Step 8 - numeric_pipeline (not yet solved)
# TODO: implement

# Step 9 - categorical_pipeline (not yet solved)
# TODO: implement

# Step 10 - build_preprocessing (not yet solved)
# TODO: implement

# Step 11 - rmse (not yet solved)
# TODO: implement

# Step 12 - dummy_baseline_rmse (not yet solved)
# TODO: implement

# Step 13 - cross_val_rmse (not yet solved)
# TODO: implement

# Step 14 - linear_model (not yet solved)
# TODO: implement

# Step 15 - forest_model (not yet solved)
# TODO: implement

# Step 16 - random_search (not yet solved)
# TODO: implement

# Step 17 - test_rmse (not yet solved)
# TODO: implement

# Step 18 - bootstrap_rmse_ci (not yet solved)
# TODO: implement

# Step 19 - feature_importances (not yet solved)
# TODO: implement

# Step 20 - worst_errors (not yet solved)
# TODO: implement

# Step 21 - save_and_reload (not yet solved)
# TODO: implement

# Step 22 - predict_new (not yet solved)
# TODO: implement

