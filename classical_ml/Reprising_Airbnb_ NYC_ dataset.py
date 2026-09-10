import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

data = pd.read_csv("https://raw.githubusercontent.com/lmassaron/tabular_datasets/master/AB_NYC_2019.csv")

# A list of features to be excluded from data processing
excluding_list = ['price', 'id', 'latitude', 'longitude', 'host_id',
                  'last_review', 'name', 'host_name']  # A

# B list of low-cardinality categorical features to be one-hot encoded
low_card_categorical = ['neighbourhood_group', 'room_type']  # B

# C list of high-cardinality categorical features to be ordinally encoded
high_card_categorical = ['neighbourhood']  # C
continuous = ['minimum_nights', 'number_of_reviews', 'reviews_per_month',
              'calculated_host_listings_count', 'availability_365']

# D creating a binary target indicating whether the price is above the mean (unbalanced binary target)
target_mean = (data["price"] > data["price"].mean()).astype(int)  # D

# E creating a binary target indicating whether the price is above the median (balanced binary target)
target_median = (data["price"] > data["price"].median()).astype(int)  # E

# F creating a multiclass target by quantile binning the price into 5 classes
target_multiclass = pd.qcut(data["price"], q=5, labels=False)  # F

# G setting the target for regression as the price column
target_regression = data["price"]  # G
categorical_onehot_encoding = OneHotEncoder(handle_unknown='ignore')
categorical_ord_encoding = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
numeric_passthrough = SimpleImputer(strategy="constant", fill_value=0)

# H creating a column transformer that applies different transformations to different groups of features
column_transform = ColumnTransformer(
    [('low_card_categories', categorical_onehot_encoding, low_card_categorical),
     ('high_card_categories', categorical_ord_encoding, high_card_categorical),
     ('numeric', numeric_passthrough, continuous),
     ],
    remainder='drop',
    verbose_feature_names_out=False,
    sparse_threshold=0.0)  # H
