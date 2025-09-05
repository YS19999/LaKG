import os
import random
import time
import datetime
from collections import OrderedDict
import itertools
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from termcolor import colored

from train.utils import named_grad_param, grad_param, get_norm

def _copy_weights(source, target):
    '''
        Copy weights from the source net to the target net
        Only copy weights with requires_grad=True
    '''
    target_dict = target.state_dict()
    for name, p in source.named_parameters():
        if p.requires_grad:
            target_dict[name].copy_(p.data.clone())

def _meta_update(model, total_grad, opt, task, maml_batchsize, clip_grad):
    '''
        Aggregate the gradients in total_grad
        Update the initialization in model
    '''

    model['clf'].train()
    support, query, label_, id2label = task["support"], task["query"], task["label"], task["id2label"]
    pred = model['clf'](support=support)
    loss = torch.sum(pred)  # this doesn't matter

    # aggregate the gradients (skip nan)
    avg_grad = {
            'clf': {key: sum(g[key] for g in total_grad['clf'] if
                        not torch.sum(torch.isnan(g[key])) > 0)\
                    for key in total_grad['clf'][0].keys()}
            }

    # register a hook on each parameter in the model that replaces
    # the current dummy grad with the meta gradiets
    hooks = []
    for model_name in avg_grad.keys():
        for key, value in model[model_name].named_parameters():
            if not value.requires_grad:
                continue

            def get_closure():
                k = key
                n = model_name
                def replace_grad(grad):
                    return avg_grad[n][k] / maml_batchsize
                return replace_grad

            hooks.append(value.register_hook(get_closure()))

    opt.zero_grad()
    loss.backward()

    clf_grad = get_norm(model['clf'])
    if clip_grad is not None:
        nn.utils.clip_grad_value_(
                grad_param(model, ['clf']), clip_grad)

    opt.step()

    for h in hooks:
        # remove the hooks before the next training phase
        h.remove()

    total_grad['clf'] = []

    return clf_grad

def train(train_data, val_data, model, args):
    '''
        Train the model (obviously~)
    '''
    # creating a tmp directory to save the models
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

    # clone the original model
    fast_model = {
            'clf': copy.deepcopy(model['clf']),
            }

    print("{}, Start training".format(
        datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')))

    for ep in range(args.train_epochs):
        random.shuffle(train_data)
        sampled_tasks = iter(train_data[:args.train_episodes])

        meta_grad_dict = {'clf': []}

        train_episodes = range(args.train_episodes)
        if not args.notqdm:
            train_episodes = tqdm(train_episodes, ncols=80, leave=False,
                                  desc=colored('Training on train', 'yellow'))

        for _ in train_episodes:
            # update the initialization based on a batch of tasks
            total_grad = {'clf': []}

            task = next(sampled_tasks)
            for _ in range(args.maml_batchsize):
                # print('start', flush=True)

                # clone the current initialization
                _copy_weights(model['clf'], fast_model['clf'])

                # get the meta gradient
                if args.maml_firstorder:
                    train_one_fomaml(task, fast_model, args, total_grad)
                else:
                    train_one(task, fast_model, args, total_grad)

            clf_grad = _meta_update(
                    model, total_grad, opt, task, args.maml_batchsize,
                    args.clip_grad)
            meta_grad_dict['clf'].append(clf_grad)

        # evaluate validation accuracy
        cur_acc, cur_std, cur_p, p_std, cur_re, cur_restd, cur_f1, cur_f1std = test(val_data, model, args, args.val_episodes, False)
        # file.writelines(str(acc) + '\t' + str(cur_acc) + '\n')
        print(("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f}, {:s}{:>7.4f} ± {:>6.4f}, {:s}{:>7.4f} ± {:>6.4f}, {:s}{:>7.4f} ± {:>6.4f},"
               "{:s}  {:s}{:>7.4f}").format(
               datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'),
               "ep", ep,
               colored("val  ", "cyan"),
               colored("acc:", "blue"), cur_acc, cur_std,
               colored("precision:", "blue"), cur_p, p_std,
               colored("recall:", "blue"), cur_re, cur_restd,
               colored("f1:", "blue"), cur_f1, cur_f1std,
               colored("train stats", "cyan"),
               colored("clf_grad:", "blue"), np.mean(np.array(meta_grad_dict['clf'])),
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

    print("{}, End of training. Restore the best weights".format(
            datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')))

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

def train_one(task, fast, args, total_grad):
    '''
        Update the fast_model based on the support set.
        Return the gradient w.r.t. initializations over the query set
    '''
    support, query, id2label = task["support"], task["query"], task["id2label"]

    # map class label into 0,...,num_classes-1
    YS = support["s_label"]
    YQ = query["q_label"]

    fast['clf'].train()

    # get weights
    fast_weights = {
        'clf': OrderedDict(
            (name, param) for (name, param) in named_grad_param(fast, ['clf'])),
        }

    num_clf_w = len(fast_weights['clf'])

    # fast adaptation
    for i in range(args.maml_innersteps):
        if i == 0:
            pred = fast['clf'](support=support)
            loss, threshold = fast['clf']._compute_loss(pred, YS)
            grads = torch.autograd.grad(loss, grad_param(fast, ['clf']), create_graph=True, allow_unused=True)
        else:
            pred = fast['clf'](support, weights=fast_weights['clf'])
            loss, threshold = fast['clf']._compute_loss(pred, YS)
            grads = torch.autograd.grad(loss, fast_weights['clf'].values(), create_graph=True)

        # update fast weight
        # fast_weights['clf'] = OrderedDict((name, param-args.maml_stepsize*grad) for ((name, param), grad) in zip(fast_weights['clf'].items(), grads))
        for ((name, param), grad) in zip(fast_weights['clf'].items(), grads):
            if grad is not None:
                fast_weights['clf'][name].data.copy_((param - args.maml_stepsize * grad).data.clone())


    # forward on the query, to get meta loss
    pred = fast['clf'](support=query, weights=fast_weights['clf'])
    loss, threshold = fast['clf']._compute_loss(pred, YQ)

    grads = torch.autograd.grad(loss, grad_param(fast, ['clf']))

    grads_clf = {name: g for ((name, _), g) in zip(
        named_grad_param(fast, ['clf']),
        grads)}

    total_grad['clf'].append(grads_clf)

    return

def train_one_fomaml(task, fast, args, total_grad):
    '''
        Update the fast_model based on the support set.
        Return the gradient w.r.t. initializations over the query set
        First order MAML
    '''
    support, query, label_, id2label = task["support"], task["query"], task["label"], task["id2label"]

    # map class label into 0,...,num_classes-1
    YS, YQ = support["s_label"], query["q_label"]

    opt = torch.optim.SGD(grad_param(fast, ['clf']),
                          lr=args.maml_stepsize)

    fast['clf'].train()

    # fast adaptation
    for i in range(args.maml_innersteps):
        opt.zero_grad()

        acc, precision, recall, f1, loss = fast['clf'](support=support, YS=YS, id2label=id2label)

        loss.backward()

        opt.step()

    # forward on the query, to get meta loss
    acc, precision, recall, f1, loss = fast['clf'](support=query, YS=YQ, id2label=id2label)

    loss.backward()
    grads_clf = {name: p.grad for (name, p) in named_grad_param(fast, ['clf'])}

    total_grad['clf'].append(grads_clf)

    return

def test(test_data, model, args, num_episodes, verbose=True, sampled_tasks=None):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    # clone the original model
    fast_model = {
            'clf': copy.deepcopy(model['clf']),
            }

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

    sampled_tasks = enumerate(sampled_tasks)
    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=num_episodes, ncols=80,
                leave=False, desc=colored('Testing on val', 'yellow'))

    for i, task in sampled_tasks:
        if i == num_episodes and not args.notqdm:
            sampled_tasks.close()
            break
        _copy_weights(model['clf'], fast_model['clf'])
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

def test_one(task, fast, args):
    '''
        Evaluate the model on one sampled task. Return the accuracy.
    '''
    support, query, label_, id2label = task["support"], task["query"], task["label"], task["id2label"]

    # map class label into 0,...,num_classes-1
    YS, YQ = support["s_label"], query["q_label"]

    fast['clf'].train()

    opt = torch.optim.SGD(grad_param(fast, ['clf']),
                          lr=args.maml_stepsize)

    for i in range(20):
        # with torch.no_grad():
        #     XS = fast['clf'].ebd(support)
        # pred = fast['clf'].mlp(XS)
        acc, precision, recall, f1, loss = fast['clf'](support=support, YS=YS, id2label=id2label)
        # pred = fast['clf'](support=support)
        # loss = F.multilabel_soft_margin_loss(pred, YS)

        opt.zero_grad()
        loss.backward()
        opt.step()

    fast['clf'].eval()

    acc, precision, recall, f1, loss = fast['clf'](support=query, YS=YQ, id2label=id2label)
    # pred = fast['clf'](support=query)
    # loss, threshold = fast['clf']._compute_loss(pred, query["q_label"])
    # pred = (pred > threshold).long()
    # acc, precision, recall, f1 = fast['clf'].compute_metric_score(pred, query["q_label"], id2label)

    return acc, precision, recall, f1
