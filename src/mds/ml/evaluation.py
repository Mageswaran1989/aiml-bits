import warnings
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

np.random.seed(0)
warnings.filterwarnings('ignore')


class Evaluator(object):
    @staticmethod
    def confusion_matrix(y_true, y_pred, labels=[1, 0], title=None):
        classes = ["B", "S"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=classes)
        fig, ax = plt.subplots(figsize=(10, 10))
        if title:
            plt.title(title)
        else:
            plt.title("Confusion Matrix")
        disp = disp.plot(ax=ax)
        fig.savefig(title + ".png")
        plt.show(block=True)

    @staticmethod
    def classification_report(y_true, y_pred, title=None):
        # print(classification_report(y_true, y_pred, labels=[1,0]))
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        print("precision_score :", p)
        print("recall_score: ", r)
        print("f1_score: ", f1)
        print("accuracy_score :", acc)

        return {title: [p, r, f1, acc]}