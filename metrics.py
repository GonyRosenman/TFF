from sklearn.metrics import balanced_accuracy_score as bac
from sklearn.metrics import roc_curve,roc_auc_score
import numpy as  np


class Metrics():
    def __init__(self):
        pass

    def BAC(self,truth,pred):
        return bac(truth,pred)

    def RAC(self,truth,pred):
        accuracy = np.sum([ii == jj for ii, jj in zip(truth, pred)]) / len(truth)
        return accuracy

    def AUROC(self,truth,pred):
        score = roc_auc_score(truth, pred)
        fpr, tpr, thresholds = roc_curve(truth, pred)
        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1 - fpr))
        # locate the index of the largest g-mean
        ix = np.argmax(gmeans)
        print('Best Threshold={}, G-Mean={}'.format(thresholds[ix], gmeans[ix]))
        return score

    def MAE(self,truth,pred):
        mae = np.mean([abs(ii - jj) for ii, jj in zip(pred, truth)])
        return mae

    def MSE(self,truth,pred):
        mse = np.mean([(ii - jj) ** 2 for ii, jj in zip(pred, truth)])
        return mse

    def NMSE(self,truth,pred):
        NMSE = np.mean([((ii - jj) ** 2) / (jj ** 2) for ii, jj in zip(pred, truth)])
        return NMSE