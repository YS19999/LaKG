import os
import random
import time
import datetime

import torch
import torch.nn as nn
import numpy as np


from tqdm import tqdm
from termcolor import colored

from train.utils import named_grad_param, grad_param, get_norm


def train(train_data, val_data, model, args):
    '''
        Train the model
        Use val_data to do early stopping
    '''

    out_dir = os.path.abspath(os.path.join(
                                  os.path.curdir,
                                  "tmp-runs",
                                  args.classifier + "_" + args.dataset + "_" + str(args.shot)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    best_acc = 0
    sub_cycle = 0
    best_path = None

    opt = torch.optim.Adam(grad_param(model, ['clf']), lr=args.lr)

    print("{}, Start training".format(
        datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')), flush=True)

    for ep in range(args.train_epochs):
        random.shuffle(train_data)
        sampled_tasks = train_data[:args.train_episodes]

        grad = {'clf': []}

        if not args.notqdm:
            sampled_tasks = tqdm(sampled_tasks, total=args.train_episodes,
                    ncols=80, leave=False, desc=colored('Training on train',
                        'yellow'))

        for task in sampled_tasks:
            if task is None:
                break
            train_one(task, model, opt, args, grad)

        # evaluate validation accuracy
        cur_acc, cur_std, cur_p, p_std, cur_re, cur_restd, cur_f1, cur_f1std = test(val_data, model, args,
                                                                                    args.val_episodes, False)

        print((
            "{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f}, {:s}{:>7.4f} ± {:>6.4f}, {:s}{:>7.4f} ± {:>6.4f}, {:s}{:>7.4f} ± {:>6.4f},"
            "{:s}  {:s}{:>7.4f}").format(
            datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'),
            "ep", ep,
            colored("val  ", "cyan"),
            colored("acc:", "blue"), cur_acc, cur_std,
            colored("precision:", "blue"), cur_p, p_std,
            colored("recall:", "blue"), cur_re, cur_restd,
            colored("f1:", "blue"), cur_f1, cur_f1std,
            colored("train stats", "cyan"),
            colored("clf_grad:", "blue"), np.mean(np.array(grad['clf'])),
        ), flush=True)

        # Update the current best model if val acc is better
        if cur_f1 > best_acc:
            best_acc = cur_f1
            best_path = os.path.join(out_dir, str(ep))

            # save current model
            print("{}, Save cur best model to {}".format(
                datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'),
                best_path))

            torch.save(model['clf'].state_dict(), best_path + '.clf')

            sub_cycle = 0
        else:
            sub_cycle += 1

        # Break if the val acc hasn't improved in the past patience epochs
        if sub_cycle == args.patience:
            break

    print("{}, End of training. Restore the best weights".format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')),
          flush=True)

    # restore the best saved model
    model['clf'].load_state_dict(torch.load(best_path + '.clf'))

    if args.save:
        # save the current model
        out_dir = os.path.abspath(os.path.join(
                                      os.path.curdir,
                                      "saved-runs",
                                      str(int(time.time() * 1e7))))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        best_path = os.path.join(out_dir, 'best')

        print("{}, Save best model to {}".format(
            datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'),
            best_path), flush=True)

        torch.save(model['clf'].state_dict(), best_path + '.clf')

        with open(best_path + '_args.txt', 'w') as f:
            for attr, value in sorted(args.__dict__.items()):
                f.write("{}={}\n".format(attr, value))

    return


def train_one(task, model, opt, args, grad):
    '''
        Train the model on one sampled task.
    '''
    model['clf'].train()
    opt.zero_grad()

    support, query, id2label = task["support"], task["query"], task["id2label"]
    if args.classifier == "dckpn":
        label = task["label"]
        loss, acc, precision, recall, f1 = model['clf'](support, query, label, id2label)
    else:
        loss, acc, precision, recall, f1 = model['clf'](support, query, id2label)

    if loss is not None:
        loss.backward()

    if torch.isnan(loss):
        # do not update the parameters if the gradient is nan
        # print("NAN detected")
        # print(model['clf'].lam, model['clf'].alpha, model['clf'].beta)
        return

    if args.clip_grad is not None:
        nn.utils.clip_grad_value_(grad_param(model, ['clf']),
                                  args.clip_grad)

    grad['clf'].append(get_norm(model['clf']))

    opt.step()


def test(test_data, model, args, num_episodes, verbose=True):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    model['clf'].eval()

    if verbose:
        episodes = args.test_episodes
    else:
        episodes = args.val_episodes
    sampled_tasks = []
    data_len = len(test_data)
    for _ in range(0, episodes):
        temp = random.randint(0, data_len-1)
        sampled_tasks.append(test_data[temp])

    acc, precision, recall, f1 = [], [], [], []
    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=len(sampled_tasks), ncols=80,
                             leave=False,
                             desc=colored('Testing on val', 'yellow'))

    for task in sampled_tasks:
        acc1, precision1, recall1, f11 = test_one(task, model, args)
        acc.append(acc1)
        precision.append(precision1)
        recall.append(recall1)
        f1.append(f11)

    acc = np.array(acc)
    precision = np.array(precision)
    recall = np.array(recall)
    f1 = np.array(f1)

    if verbose:
        print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'),
                colored("acc mean", "blue"),
                np.mean(acc),
                colored("acc std", "blue"),
                np.std(acc),
                colored("precision mean", "blue"),
                np.mean(precision),
                colored("precision std", "blue"),
                np.std(precision),
                colored("recall mean", "blue"),
                np.mean(recall),
                colored("recall std", "blue"),
                np.std(recall),
                colored("f1 mean", "blue"),
                np.mean(f1),
                colored("f1 std", "blue"),
                np.std(f1),
                ), flush=True)

    return np.mean(acc), np.std(acc), np.mean(precision), np.std(precision), np.mean(recall), np.std(recall), np.mean(f1), np.std(f1)


def test_one(task, model, args):
    '''
        Evaluate the model on one sampled task. Return the accuracy.
    '''
    model['clf'].train()

    support, query, id2label = task["support"], task["query"], task["id2label"]
    if args.classifier == "dckpn":
        label = task["label"]
        loss, acc, precision, recall, f1 = model['clf'](support, query, label, id2label)
    else:
        loss, acc, precision, recall, f1 = model['clf'](support, query, id2label)

    return acc, precision, recall, f1


