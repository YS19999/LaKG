import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

from classifier.base import BASE

class MLP1(BASE):
    def __init__(self, ebd, args):
        super(MLP1, self).__init__(args, args.threshold, False)
        self.ebd = ebd

        self.ebd_dim = self.ebd.ebd_dim

        if args.dataset in ['it', 'ac', 'at', 'fo', 'tr', 'sh']:
            num_classes = 19
        else:
            num_classes = 24
        self.num_classes = num_classes
        self.r = nn.Linear(self.ebd_dim, num_classes, bias=True)
        nn.init.kaiming_normal_(self.r.weight, a=0, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.r.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.la2dt = nn.Sequential(
            nn.Linear(num_classes, self.ebd_dim)
        )

        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.ebd_dim, self.ebd_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.ebd_dim, self.ebd_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.ebd_dim, num_classes),
        )

        self.la_loss = nn.MultiLabelSoftMarginLoss()

    def Adaptive_knowledge(self, prototype):

        C = prototype
        eps = 1e-6
        R = self.r.weight

        power_R = ((R * R).sum(dim=1, keepdim=True)).sqrt()
        R = R / (power_R + eps)

        power_C = ((C * C).sum(dim=1, keepdim=True)).sqrt()
        C = C / (power_C + eps)

        P = torch.matmul(torch.pinverse(C), R)
        P = P.permute(1, 0)
        return P

    def forward(self, support, query=None, label=None, YS=None, YQ=None, id2label=None):

        XS = self.ebd(support)
        prototype = self.ebd(label)
        P = self.Adaptive_knowledge(prototype)
        weight = P.view(P.size(0), P.size(1), 1)
        prototype = F.conv1d(prototype.squeeze(0).unsqueeze(2), weight).squeeze(2)
        XS = F.conv1d(XS.squeeze(0).unsqueeze(2), weight).squeeze(2)

        if query is not None:
            XQ = self.ebd(query)
            XQ = F.conv1d(XQ.squeeze(0).unsqueeze(2), weight).squeeze(2)
            all_x = torch.cat([XS, XQ], dim=0)
            all_label = torch.cat([support["s_label"], query["q_label"]], dim=0)

            la2dt = self.la2dt(all_label.float())
            la2dt = F.conv1d(la2dt.squeeze(0).unsqueeze(2), weight).squeeze(2)
            la_logits = torch.sigmoid(self.mlp(la2dt))

            la_loss = self.la_loss(la_logits, all_label.float())

            discriminative_loss = 0.0
            for j in range(self.num_classes):
                for k in range(self.num_classes):
                    if j != k:
                        sim = -self._compute_cos(prototype[j].unsqueeze(0), prototype[k].unsqueeze(0))
                        discriminative_loss = discriminative_loss + sim

            logits = self.mlp(all_x)
            loss, threshold = self._compute_loss(logits, all_label)
            loss = loss + 0.3 * discriminative_loss + la_loss
            pred = (logits > self.threshold).long()
            acc, precision, recall, f1 = self.compute_metric_score(pred, all_label, id2label)
            return acc, precision, recall, f1, loss

        else:
            return self.mlp(XS), None

    def inference_ada(self, support, label):

        XS = self.ebd(support)
        prototype = self.ebd(label)
        P = self.Adaptive_knowledge(prototype)
        weight = P.view(P.size(0), P.size(1), 1)
        prototype = F.conv1d(prototype.squeeze(0).unsqueeze(2), weight).squeeze(2)
        XS = F.conv1d(XS.squeeze(0).unsqueeze(2), weight).squeeze(2)

        la2dt = self.la2dt(support['s_label'].float())
        la2dt = F.conv1d(la2dt.squeeze(0).unsqueeze(2), weight).squeeze(2)
        la_logits = torch.sigmoid(self.mlp(la2dt))

        la_loss = self.la_loss(la_logits, support['s_label'].float())

        discriminative_loss = 0.0
        for j in range(self.args.way):
            for k in range(self.args.way):
                if j != k:
                    sim = -self._compute_cos(prototype[j].unsqueeze(0), prototype[k].unsqueeze(0))
                    discriminative_loss = discriminative_loss + sim

        logits = self.mlp(XS)
        loss, threshold = self._compute_loss(logits, support['s_label'])
        loss = loss + 0.3 * discriminative_loss + la_loss
        return loss