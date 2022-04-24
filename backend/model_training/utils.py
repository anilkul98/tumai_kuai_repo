import numpy as np

class ConfusionMatrixTracker:

    def __init__(self, num_classes=2, class_names=None):
        self.num_classes = num_classes
        self.class_names = [str(i) for i in range(num_classes)]\
            if class_names is None else class_names
        assert len(self.class_names) == num_classes, "Number of class names should be equal to num_classes"
        self.cm = np.zeros((num_classes, num_classes))
        self.y_trues = []
        self.y_preds = []

    def reset_states(self):
        self.cm = np.zeros((self.num_classes, self.num_classes))
        self.y_trues = []
        self.y_preds = []

    def update_state(self, y_true, y_pred):
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.cm[j, i] += (y_true[y_pred == i] == j).sum()
        self.y_trues.extend(y_true)
        self.y_preds.extend(y_pred)

    def calculate_precision(self, class_nr):
        prec = self.cm[class_nr, class_nr] / self.cm[:, class_nr].sum()
        return 0 if np.isnan(prec) else prec

    def calculate_recall(self, class_nr):
        recall = self.cm[class_nr, class_nr] / self.cm[class_nr, :].sum()
        return 0 if np.isnan(recall) else recall

    def calculate_f1(self, class_nr):
        recall = self.calculate_recall(class_nr)
        prec = self.calculate_precision(class_nr)
        return ConfusionMatrixTracker.calculate_f1_from_rp(recall, prec)

    @staticmethod
    def calculate_f1_from_rp(recall, prec):
        return (2 * prec * recall) / (prec + recall + 1e-6)

    def calculate_accuracy(self):
        return self.cm.diagonal().sum() / self.cm.sum()

    def log_metrics(self):
        metrics = {}
        metrics["Accuracy"] = self.calculate_accuracy()
        for i, name in enumerate(self.class_names):
            recall = self.calculate_recall(i)
            prec = self.calculate_precision(i)
            metrics[name + " Recall"] = recall
            metrics[name + " Precision"] = prec
            metrics[name + " F1"] = ConfusionMatrixTracker.calculate_f1_from_rp(recall, prec)
        metrics["Average F1"] = np.mean([v for k, v in metrics.items() if k.endswith(" F1")])
        return metrics