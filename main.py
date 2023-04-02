import sys
import os.path as osp

SRC_SUBDIR = 'src/'
SRC_SUBDIR = osp.abspath(SRC_SUBDIR)
if SRC_SUBDIR not in sys.path:
    print(f'Adding source directory to the sys.path: {SRC_SUBDIR!r}')
    sys.path.insert(1, SRC_SUBDIR)

import pandas as pd
from sklearn.model_selection import train_test_split
from mds.ml.dataset import MalwareDataset
from mds.ml import pipelines as mds_pipelines
from mds.ml.evaluation import Evaluator
from joblib import dump

if __name__ == "__main__":
    dataset = MalwareDataset(dataset_url="data/drebin-215-dataset-5560malware-9476-benign.csv",
                             dataset_info_url="data/dataset-features-categories.csv")
    target_col = "class"
    feature_cols = dataset._get_feature_cols(dataset.malware_df)
    target_array = dataset.get_target_array(dataset.malware_df)

    df = dataset.malware_df
    train_x, test_x, train_y, test_y = train_test_split(
                    df[feature_cols],
                    df[target_col],
                    test_size=0.2,
                    shuffle=True)

    # ------------------------------------------------------------------------------------------------------------------

    evaluation_results = {}

    res = mds_pipelines.rf_vanilla_clf_pipeline.fit(train_x, train_y)
    y_pred = res.predict(test_x)
    Evaluator.confusion_matrix(y_true=test_y, y_pred=y_pred, title="RandomForest with vanilla features")
    evaluation_results.update(
        Evaluator.classification_report(y_true=test_y, y_pred=y_pred, title="RandomForest with vanilla features")
    )

    res = mds_pipelines.rf_corr_engd_clf_pipeline.fit(train_x, train_y)
    y_pred = res.predict(test_x)
    Evaluator.confusion_matrix(y_true=test_y, y_pred=y_pred, title="RandomForest with correlation corrected features")
    evaluation_results.update(
        Evaluator.classification_report(y_true=test_y, y_pred=y_pred, title="RandomForest with correlation corrected features")
    )
    print("Storing the efficient model...")
    dump(mds_pipelines.rf_corr_engd_clf_pipeline, "data/model_store/rf_corr_engd_clf_pipeline.joblib")

    res = mds_pipelines.rf_chi_square_engd_clf_pipeline.fit(train_x, train_y)
    y_pred = res.predict(test_x)
    Evaluator.confusion_matrix(y_true=test_y, y_pred=y_pred, title="RandomForest with Chi Square features")
    evaluation_results.update(
        Evaluator.classification_report(y_true=test_y, y_pred=y_pred, title="RandomForest with Chi Square features")
    )

    # ------------------------------------------------------------------------------------------------------------------

    res = mds_pipelines.svc_vanilla_clf_pipeline.fit(train_x, train_y)
    y_pred = res.predict(test_x)
    Evaluator.confusion_matrix(y_true=test_y, y_pred=y_pred, title="SVC with vanilla features")
    evaluation_results.update(
        Evaluator.classification_report(y_true=test_y, y_pred=y_pred, title="SVC with vanilla features")
    )

    res = mds_pipelines.svc_corr_engd_clf_pipeline.fit(train_x, train_y)
    y_pred = res.predict(test_x)
    Evaluator.confusion_matrix(y_true=test_y, y_pred=y_pred, title="SVC with correlation corrected features")
    evaluation_results.update(
        Evaluator.classification_report(y_true=test_y, y_pred=y_pred, title="SVC with correlation corrected features")
    )

    res = mds_pipelines.svc_chi_square_engd_clf_pipeline.fit(train_x, train_y)
    y_pred = res.predict(test_x)
    Evaluator.confusion_matrix(y_true=test_y, y_pred=y_pred, title="SVC with Chi Square features")
    evaluation_results.update(
        Evaluator.classification_report(y_true=test_y, y_pred=y_pred, title="SVC with Chi Square features")
    )

    # ------------------------------------------------------------------------------------------------------------------

    evaluation_results_df = pd.DataFrame(evaluation_results, index=["Precision", "Recall", "F1", "Accuracy"])
    evaluation_results_df.style.highlight_max(color='red', axis=1)


