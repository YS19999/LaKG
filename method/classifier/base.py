import numpy as np
import torch
import torch.nn as nn

CUDA_LAUNCH_BLOCKING=1


class BASE(nn.Module):

    def __init__(self, args, threshold=0.6, grad_threshold=True):
        super(BASE, self).__init__()
        self.args = args
        self.threshold = nn.Parameter(torch.FloatTensor([threshold]), requires_grad=grad_threshold)
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.right_estimate = None

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        :param logits: [query, class_num]
        :param targets: [query, class_num]
        :return:
        """

        threshold = self.get_threshold(logits) # [q, 1]
        filtered_logits = logits - threshold
        loss = self.criterion(filtered_logits.float(), targets.float())
        return loss, threshold

    def get_threshold(self, logits):
        return self.threshold

    def _compute_l2(self, XS, XQ):
        """
            Compute the pairwise l2 distance
            @param XS (support x): support_size x ebd_dim
            @param XQ (support x): query_size x ebd_dim

            @return dist: query_size x support_size

        """
        diff = XS.unsqueeze(0) - XQ.unsqueeze(1)
        dist = torch.norm(diff, dim=2)

        return dist

    def _compute_cos(self, XS, XQ):
        """
            Compute the pairwise cos distance
            @param XS (support x): support_size x ebd_dim
            @param XQ (support x): query_size x ebd_dim

            @return dist: query_size support_size

        """
        dot = torch.matmul(
            XS.unsqueeze(0).unsqueeze(-2),
            XQ.unsqueeze(1).unsqueeze(-1)
        )
        dot = dot.squeeze(-1).squeeze(-1)

        scale = (
                torch.norm(XS, dim=1).unsqueeze(0) *
                torch.norm(XQ, dim=1).unsqueeze(1)
        )

        scale = torch.max(scale, torch.ones_like(scale) * 1e-8)

        dist = 1 - dot/scale

        return dist

    @staticmethod
    def compute_acc(pred, true):
        acc = 0
        for p, t in zip(pred, true):
            if p.tolist() == t.tolist():
                acc += 1
        return acc / true.shape[0]

    def compute_metric_score(self, pred, true, id2label):
        pred1 = []
        for line in pred:
            indices = torch.nonzero(line).squeeze(-1)
            temp = []
            for idx in indices:
                temp.append(id2label[idx.item()])
            pred1.append(temp)

        true1 = []
        for line in true:
            indices = torch.nonzero(line).squeeze(-1)
            temp = []
            for idx in indices:
                temp.append(id2label[idx.item()])
            true1.append(temp)

        acc = self.compute_acc(pred, true)

        tp, fn, fp, tn = self.some_samples(true1, pred1)
        recall = (tp + 1e-7) / (tp + fn + 1e-7)
        precision = (tp + 1e-7) / (tp + fp + 1e-7)
        # print(tp, fn, fp, tn, recall, precision)
        f1 = 2 * recall * precision / (precision + recall)

        return acc, precision, recall, f1

    def some_samples(self, y_trues, y_preds):
        """ 评估多个样本的TP、FN、FP、TN """
        if len(y_trues) == len(y_preds):
            tp = 0
            fn = 0
            fp = 0
            tn = 0
            for i in range(len(y_trues)):
                y_true = y_trues[i]
                y_pred = y_preds[i]
                tpi, fni, fpi, tni = self.single_sample(y_true, y_pred)
                tp = tp + tpi
                fn = fn + fni
                fp = fp + fpi
                tn = tn + tni
            return tp, fn, fp, tn
        else:
            print('Different length between y_trues and y_preds!')
            return 0, 0, 0, 0

    def single_sample(self, y_true, y_pred):
        """ 评估单个样本的TP、FN、FP、TN """
        y_true = list(set(y_true))
        y_pred = list(set(y_pred))
        y_ = list(set(y_true) | set(y_pred))
        K = len(y_)
        tp1 = 0
        fn1 = 0
        fp1 = 0
        tn1 = 0
        for i in range(len(y_)):
            if y_[i] in y_true and y_[i] in y_pred:
                tp1 = tp1 + 1 / K
            elif y_[i] in y_true and y_[i] not in y_pred:
                fn1 = fn1 + 1 / K
            elif y_[i] not in y_true and y_[i] in y_pred:
                fp1 = fp1 + 1 / K
            elif y_[i] not in y_true and y_[i] not in y_pred:
                tn1 = tn1 + 1 / K
        return tp1, fn1, fp1, tn1




