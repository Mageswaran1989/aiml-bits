import warnings
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
np.random.seed(0)


class MalwareDataset(object):
    dataset_url = "https://raw.githubusercontent.com/Mageswaran1989/aiml-bits/master/drebin-215-dataset-5560malware-9476-benign.csv"
    dataset_info_url = "https://raw.githubusercontent.com/Mageswaran1989/aiml-bits/master/dataset-features-categories.csv"

    def __init__(self, dataset_url=None, dataset_info_url=None, target_col="class"):

        self.target_col = target_col

        if dataset_url:
            self.dataset_features = pd.read_csv(dataset_info_url, header=None)
        else:
            self.dataset_features = pd.read_csv(MalwareDataset.dataset_info_url, header=None)

        if dataset_info_url:
            self.malware_df = pd.read_csv(dataset_url)
        else:
            self.malware_df = pd.read_csv(MalwareDataset.dataset_url)

        self.label_encoding()
        self.clean()

    def clean(self):
        self.malware_df = self.malware_df.replace('[?,S]', np.NaN, regex=True)
        print("Total missing values : ", sum(list(self.malware_df.isna().sum())))

        self.malware_df.dropna(inplace=True)

        for c in self.malware_df.columns:
            self.malware_df[c] = pd.to_numeric(self.malware_df[c])

    def _get_feature_cols(self, df=None):
        if df is None:
            df = self.malware_df
        return [col for col in df.columns if col != "class"]

    def get_target_array(self, df):
        return df[self.target_col].to_numpy()

    def get_numpy_data(self, df):
        train_x, test_x, train_y, test_y = train_test_split(
            df[df.columns[:len(df.columns) - 1]].to_numpy(),
            df[df.columns[-1]].to_numpy(),
            test_size=0.2,
            shuffle=True)

        train_y = train_y.reshape((-1, 1))
        test_y = test_y.reshape((-1, 1))

        train_x = train_x.astype(np.float32)
        test_x = test_x.astype(np.float32)
        train_y = train_y.astype(np.float32)
        test_y = test_y.astype(np.float32)

        print("Train features : ", train_x.shape)
        print("Train labels : ", train_y.shape)
        print("Test Features : ", test_x.shape)
        print("Test labels : ", test_y.shape)

        return train_x, train_y, test_x, test_y

    def plot_classes(self):
        y = self.malware_df['class']
        total = float(len(y))
        ax = sns.countplot(x=y)

        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 100, "{:1.0f}%".format((height / total) * 100),
                    ha="center")

        ax.set_title("Class Distribution")

    def plot_correlation(self):
        feature_cols = self._get_feature_cols(df=self.malware_df)
        corr = self.malware_df.loc[:, feature_cols].corr()

        sns.heatmap(corr, annot=False)

    # Category labels to numeric
    def label_encoding(self):
        # classes, count = np.unique(self.malware_df['class'], return_counts=True)
        lbl_enc = LabelEncoder()
        # print(classes, "--->", lbl_enc.fit_transform(classes))
        # self.malware_df = self.malware_df.replace(classes, lbl_enc.fit_transform(classes))

        self.malware_df[self.target_col] = lbl_enc.fit_transform(self.malware_df[self.target_col])
