import torch
from classifier.cxtebd import CXTEBD
from classifier.lakg import MLP1

from dataset.utils import tprint


def get_classifier(args):
    tprint("Building classifier: {}".format(args.classifier))

    ebd = CXTEBD(args, return_seq=False)

    model = MLP1(ebd, args)

    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model
