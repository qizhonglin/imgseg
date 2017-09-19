import numpy as np
from sklearn import metrics


class Metrics(object):
    @staticmethod
    def cvt2_bin(y_pred):
        thresh = (np.max(y_pred) - np.min(y_pred)) / 2
        y_pred[y_pred < thresh] = 0
        y_pred[y_pred >= thresh] = 1
        return y_pred

    def __init__(self, y_true, y_pred):
        self.y_true = np.asarray(Metrics.cvt2_bin(y_true)).astype(np.bool)
        self.y_pred =np.asarray(Metrics.cvt2_bin(y_pred)).astype(np.bool)

        if self.y_true.shape != self.y_pred.shape:
            raise ValueError("Shape mismatch: y_true and y_pred must have the same shape.")

    def sensitivity_specificity(self):
        # Compute TP, FP, TN, FN
        TP = np.sum(np.logical_and(self.y_pred == True, self.y_true == True))
        TN = np.sum(np.logical_and(self.y_pred == False, self.y_true == False))
        FP = np.sum(np.logical_and(self.y_pred == True, self.y_true == False))
        FN = np.sum(np.logical_and(self.y_pred == False, self.y_true == True))

        sensitivity = TP / float(TP + FN)
        specificity = TN / float(TN + FP)

        return (sensitivity, specificity)

    def accuracy(self):
        y_true = np.reshape(self.y_true, [-1])
        y_pred = np.reshape(self.y_pred, [-1])
        return metrics.accuracy_score(y_true, y_pred)

    def dice(self, smooth=0.0):
        intersection = np.logical_and(self.y_true, self.y_pred)
        dice = 2. * (intersection.sum() + smooth) / (self.y_true.sum() + self.y_pred.sum() + smooth)

        return dice

    def jaccard_similarity_score(self):
        y_true = np.reshape(self.y_true, [-1])
        y_pred = np.reshape(self.y_pred, [-1])
        return metrics.jaccard_similarity_score(y_true, y_pred)

    @staticmethod
    def all(y_trues, y_preds):
        sensitivity = 0
        specificity = 0
        acc = 0
        dice = 0
        jacc = 0
        for (y_true, y_pred) in zip(y_trues, y_preds):
            (sen, spec) = Metrics(y_true, y_pred).sensitivity_specificity()
            sensitivity += sen
            specificity += spec

            acc += Metrics(y_true, y_pred).accuracy()

            dice += Metrics(y_true, y_pred).dice()

            jacc += Metrics(y_true, y_pred).jaccard_similarity_score()

        len_y = len(y_trues)
        return {
            'sensitivity': sensitivity / len_y,
            'specificity': specificity / len_y,
            'acc': acc / len_y,
            'dice': dice / len_y,
            'jacc': jacc / len_y
        }

if __name__ == '__main__':
    pass
