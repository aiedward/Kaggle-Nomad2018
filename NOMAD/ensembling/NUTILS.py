import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, make_scorer

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
        
# Utility functions
def load_data(csv_path):
    return pd.read_csv(csv_path)

#def rmsle(y, y0):
#    return tf.sqrt(tf.reduce_mean(tf.pow(tf.log1p(y)-tf.log1p(y0), 2)))
def rmsle(actual, predicted):
    """
    Args:
        actual (1d-array [nx1]) - array of actual values (float)
        predicted (1d-array [nx1]) - array of predicted values (float)
    Returns:
        root mean square log error (float)
    """
    return np.sqrt(np.mean(np.power(np.log1p(actual)-np.log1p(predicted), 2)))

def display_rmsle(y_pred_f, y_pred_b, y_f, y_b):
    rmsle_f = rmsle(y_f, y_pred_f)
    rmsle_b = rmsle(y_b, y_pred_b)
    print ("RMSLE: ", (rmsle_f + rmsle_b) / 2)

objective  = make_scorer(rmsle, greater_is_better=False)

def drop_features(df_t):
    df = df_t.copy()
    df = df.drop('id', 1)
    df = df.drop("formation_energy_ev_natom", 1)
    df = df.drop("bandgap_energy_ev", 1)
    return df

def drop_features_s(df_t):
    df = df_t.copy()
    df = df.drop('id', 1)
    return df

def drop_features_submt(df_t):
    df = df_t.copy()
    df = df.drop('id', 1)
    return df

def display_scores(scores):
    print("Expected LB")
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())
    
def encode_scale(df):
    X = df
    encoded_matrix = OneHotEncoder(sparse=False).fit_transform(X['spacegroup'].values.reshape(-1,1))
    X = X.drop("spacegroup", 1)
    encoded_df = pd.DataFrame(data=encoded_matrix, dtype=np.float64)
    X_encoded = pd.concat([X, encoded_df], axis=1).reindex()
    print(X_encoded)
    X_scaled = StandardScaler().fit_transform(X_encoded)
    
    myEncoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    myEncoder.fit(df[columnsToEncode])
    pd.concat([X, pd.DataFrame(myEncoder.transform(df[columnsToEncode]))], axis=1).reindex()
    return X_scaled

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
def encode(df):
    numerical_attrbs = list(df)
    del numerical_attrbs[0]

    label_attrbs = ['spacegroup']
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import FeatureUnion
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
    data_prepared = full_pl.fit_transform(df)
    return pd.DataFrame(data_prepared.toarray())
