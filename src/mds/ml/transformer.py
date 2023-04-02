import warnings
from pandas import DataFrame
import numpy as np

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.base import BaseEstimator, TransformerMixin

np.random.seed(0)
warnings.filterwarnings('ignore')


class DataFrameToNumpyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col="class"):
        self.target_col = target_col
        
    def _gettarget_cols(self, df):
        return [col for col in df.columns if col != self.target_col]
    
    def get_numpy_data(self, df_x, df_y=None):
        feature_cols = self._gettarget_cols(df_x)

        train_x = df_x[feature_cols].to_numpy()
        if df_y is not None:
            if type(df_y) == DataFrame:
                train_y = df_y[self.target_col].to_numpy()
            else:
                train_y = df_y.to_numpy()

        train_x = train_x.astype(np.int32)
        if df_y is not None:
            train_y = train_y.astype(np.int32)
            train_y = train_y.reshape((-1, 1))

        print("Train features : ", train_x.shape)
        if df_y is not None:
            print("Train labels : ", train_y.shape)

        if df_y is not None:
            return train_x, train_y
        else:
            return train_x

    def fit(self, X, Y=None):
        return self

    def transform(self, X, y=None):
        return self.get_numpy_data(df_x=X)

    def fit_transform(self, X, y=None):
        print("numpy fit_transform")
        X, y = self.get_numpy_data(df_x=X, df_y=y)
        return X


class CorrelationColumnsSelector(BaseEstimator, TransformerMixin):
    # initializer
    def __init__(self, corr=0.85, target_col="class"):
        # save the features list internally in the class
        self._corr = corr
        self.target_col = target_col

        self.less_correlated_features = None

    def remove_highly_correlated_features(self, df):
        feature_cols = [col for col in df.columns if col != self.target_col]
        corr = df.loc[:, feature_cols].corr()

        highly_correlated_cols = []  # highly cor-related columns

        for col in feature_cols:
            if col not in highly_correlated_cols:
                corr_cols = corr.loc[corr[col] >= 0.85, col].index.tolist()
                corr_cols = [x for x in corr_cols if x != col]
                if len(corr_cols) > 0:
                    highly_correlated_cols += corr_cols

        print(f" There are {len(set(highly_correlated_cols))} features that are highly correlated")

        self.less_correlated_features = [col for col in df[feature_cols].columns if
                                    col not in highly_correlated_cols]
        #df = df[less_correlated_features + ["class"]]

        #return df

    def fit(self, X, y=None):
        self.remove_highly_correlated_features(df=X)
        return self

    def transform(self, X, y=None):
        # return the dataframe with the specified features
        return X[self.less_correlated_features]


class ChiSquareColumnsSelector(BaseEstimator, TransformerMixin):
    # initializer
    def __init__(self, k=128, target_col="class"):
        # save the features list internally in the class
        self._k = k
        self.target_col = target_col

        self._top_k_features = None

        self._chi2_features = SelectKBest(chi2, k=k)

    def _gettarget_cols(self, df):
        return [col for col in df.columns if col != self.target_col]

    # Feature Engineering
    def _fit_chi_square(self, df, y, k=25):
        feature_cols = self._gettarget_cols(df=df)
        self._chi2_features.fit(df[feature_cols], y)
        self._top_k_features = [df.columns.tolist()[i] for i in
                               iter(self._chi2_features.get_support(indices=True))]
        #return df[toptarget_columns + ["class"]]

    def fit(self, X, y=None):
        self._fit_chi_square(df=X, y=y)
        return self

    def transform(self, X, y=None):
        # return the dataframe with the specified features
        return X[self._top_k_features]