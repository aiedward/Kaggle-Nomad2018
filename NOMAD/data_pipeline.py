import os
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

def load_data(csv_path):
    return pd.read_csv(csv_path)
    
TRAIN_PATH = "/home/agi/Desktop/NOMAD/train.csv"
TEST_PATH = "/home/agi/Desktop/NOMAD/test.csv"

trainData = load_data(TRAIN_PATH)
testData = load_data(TEST_PATH)

# Drop duplicates by id
dups = [394,125,1214,1885,2074,352,307,2153,530,1378,2318,2336,2369,2332]
trainData = trainData.drop(trainData.index[[dups]])

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(trainData, test_size=0.2, random_state=42)

train = train_set.copy()
train = train.drop('id', 1)
train = train.drop("formation_energy_ev_natom", 1)
train = train.drop("bandgap_energy_ev", 1)
form_labels = trainData["formation_energy_ev_natom"].copy()
band_labels = trainData["bandgap_energy_ev"].copy()

numerical_attrbs = list(train)
del numerical_attrbs[0]
del numerical_attrbs[0]
label_attrbs = ['spacegroup']

from sklearn.preprocessing import OneHotEncoder
spacegroup = trainData["spacegroup"]
encoder = OneHotEncoder()
spacegroup_1hot = encoder.fit_transform(spacegroup.values.reshape(-1,1))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

attrs = list(trainData)
hot_attrs = ["spacegroup"]

pl1 = Pipeline([
    ('selector', DataFrameSelector(numerical_attrbs)),
    ('scaler', StandardScaler()),
])

pl2 = Pipeline([
    ('selector', DataFrameSelector(label_attrbs)),
    ('one_hot', OneHotEncoder())
])

full_pl = FeatureUnion(transformer_list=[
    ('numerical', pl1),
    ('label',     pl2),
])

data_prepared = full_pl.fit_transform(trainData)
print(type(data_prepared))
df = pd.DataFrame(data=data_prepared.toarray())
df.to_csv("/home/agi/Desktop/NOMAD/data_scaled_1hot.csv")
