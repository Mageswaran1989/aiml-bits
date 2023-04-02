import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from mds.ml.transformer import CorrelationColumnsSelector, ChiSquareColumnsSelector, DataFrameToNumpyTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
np.random.seed(0)

rf_vanilla_clf = [("to_numpy", DataFrameToNumpyTransformer()),
                  ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features="auto"))]
# rf_vanilla_clf = [('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features="auto"))]
rf_vanilla_clf_pipeline = Pipeline(rf_vanilla_clf)


rf_correlation_engd_clf = [('corr', CorrelationColumnsSelector(corr=0.85)),
                           ("to_numpy", DataFrameToNumpyTransformer()),
                           ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features="auto"))]
rf_corr_engd_clf_pipeline = Pipeline(rf_correlation_engd_clf)


rf_chi_square_engd_clf = [('chisqr', ChiSquareColumnsSelector(k=128)),
                           ("to_numpy", DataFrameToNumpyTransformer()),
                           ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features="auto"))]
rf_chi_square_engd_clf_pipeline = Pipeline(rf_chi_square_engd_clf)



# ----------------------------------------------------------------------------------------------------------------------

svc_vanilla_clf = [("to_numpy", DataFrameToNumpyTransformer()),
                  ('clf', SVC(C=3, gamma="scale", kernel="rbf", random_state=42))]
svc_vanilla_clf_pipeline = Pipeline(svc_vanilla_clf)


svc_correlation_engd_clf = [("corr", CorrelationColumnsSelector(corr=0.85)),
                            ("to_numpy", DataFrameToNumpyTransformer()),
                            ("clf", SVC(C=3, gamma="scale", kernel="rbf", random_state=42))]
svc_corr_engd_clf_pipeline = Pipeline(svc_correlation_engd_clf)


svc_chi_square_engd_clf = [('chisqr', ChiSquareColumnsSelector(k=128)),
                           ("to_numpy", DataFrameToNumpyTransformer()),
                           ('clf', SVC(C=3, gamma="scale", kernel="rbf", random_state=42))]
svc_chi_square_engd_clf_pipeline = Pipeline(svc_chi_square_engd_clf)