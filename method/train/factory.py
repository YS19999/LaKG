import train.maml as maml
import train.regular as regular
import train.trainer as trainer


def train(train_data, val_data, model, args):
    return trainer.train(train_data, val_data, model, args)


def test(test_data, model, args, verbose=True):

    return trainer.test(test_data, model, args, verbose)
