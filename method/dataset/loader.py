import json

import numpy as np
import pandas as pd
import torch
from dataset.utils import tprint
import dataset.masked as  masked

from transformers import BertTokenizer, BertForMaskedLM


def _load_stanford_label():
    label = {'request_low_temperature': "request low temperature", 'request_time': "request time",
             'appreciate': "appreciate", 'request_temperature': "request temperature",
             'request_weather': "request weather", 'inform': "inform",
             'request_high_temperature': "request high temperature", 'query': 'query',
             'request_poi': 'request position',
             'request_traffic': "request traffic", 'request_address': 'request address',
             'request_route': 'request route', 'confirm': 'confirm', 'show_in_screen': 'show in screen',
             'navigate': 'navigate', 'schedule': 'schedule', 'command_appointment': 'command appointment',
             'remind': 'remind', 'request_information': 'request information',
             'list_schedule': 'list schedule', 'request_party': 'request party', 'request_agenda': 'request agenda',
             'request_location': 'request location', 'request_date': 'request date'}

    return label


def _load_toursg_label():
    label = {'POSITIVE': 'POSITIVE', 'WHO': 'WHO', 'INFO': 'INFO', 'WHAT': 'WHAT', 'HOW_MUCH': 'HOW MUCH',
             'RECOMMEND': 'RECOMMEND', 'CONFIRM': 'CONFIRM',
             'WHICH': 'WHICH', 'THANK': 'THANK', 'COMMIT': 'COMMIT', 'NEGATIVE': 'NEGATIVE', 'EXPLAIN': 'EXPLAIN',
             'ACK': 'ACK', 'PREFERENCE': 'PREFERENCE',
             'WHEN': 'WHEN', 'HOW_TO': 'HOW TO', 'WHERE': 'WHERE', 'ENOUGH': 'ENOUGH', 'OPENING': 'OPENING'}

    return label


def _load_cwoz_label():
    label = {
        "推荐景点": "推荐景点",
        "询问电话": "询问电话",
        "询问地址": "询问地址",
        "询问时间": "询问时间",
        "询问周边": "询问周边",
        "询问票价": "询问票价",
        "询问评分": "询问评分",
        "问好": "问好",
        "询问服务": "询问服务",
        "询问设施": "询问设施",
        "推荐酒店": "推荐酒店",
        "询问类型": "询问类型",
        "询问价格": "询问价格",
        "推荐餐馆": "推荐餐馆",
        "推荐菜": "推荐菜",
        "询问人均消费": "询问人均消费"
    }

    return label


def _load_jwoz_label():
    label = {
        "宿泊を探": "宿泊を探",
        "時間": "時間",
        "電話": "電話",
        "住所を/エリア": "住所を エリア",
        "サービス": "サービス",
        "り駅": "り駅",
        "予算/値段": "予算 値段",
        "わかりました/はい": "わかりました はい",
        "予約": "予約",
        "ありがとうござい": "ありがとうござい",
        "レストラン/飲食": "レストラン 飲食",
        "ショッピング": "ショッピング"
    }

    return label


def _load_json(args):
    if args.dataset == "it":
        if args.shot == 1:
            path = "../data/toursg/toursg.0.spt_s_1.q_s_16.ep_100--use_schema--label_num_schema2/"
        else:
            path = "../data/toursg/toursg.0.spt_s_5.q_s_16.ep_100--use_schema--label_num_schema2/"
    elif args.dataset == "ac":
        if args.shot == 1:
            path = "../data/toursg/toursg.1.spt_s_1.q_s_16.ep_100--use_schema--label_num_schema2/"
        else:
            path = "../data/toursg/toursg.1.spt_s_5.q_s_16.ep_100--use_schema--label_num_schema2/"
    elif args.dataset == "at":
        if args.shot == 1:
            path = "../data/toursg/toursg.2.spt_s_1.q_s_16.ep_100--use_schema--label_num_schema2/"
        else:
            path = "../data/toursg/toursg.2.spt_s_5.q_s_16.ep_100--use_schema--label_num_schema2/"
    elif args.dataset == "fo":
        if args.shot == 1:
            path = "../data/toursg/toursg.3.spt_s_1.q_s_16.ep_100--use_schema--label_num_schema2/"
        else:
            path = "../data/toursg/toursg.3.spt_s_5.q_s_16.ep_100--use_schema--label_num_schema2/"
    elif args.dataset == "sh":
        if args.shot == 1:
            path = "../data/toursg/toursg.4.spt_s_1.q_s_16.ep_100--use_schema--label_num_schema2/"
        else:
            path = "../data/toursg/toursg.4.spt_s_5.q_s_16.ep_100--use_schema--label_num_schema2/"
    elif args.dataset == "tr":
        if args.shot == 1:
            path = "../data/toursg/toursg.5.spt_s_1.q_s_16.ep_100--use_schema--label_num_schema2/"
        else:
            path = "../data/toursg/toursg.5.spt_s_5.q_s_16.ep_100--use_schema--label_num_schema2/"
    elif args.dataset == "sc":
        if args.shot == 1:
            path = "../data/stanford/stanford.0.spt_s_1.q_s_32.ep_200--use_schema--label_num_schema2/"
        else:
            path = "../data/stanford/stanford.0.spt_s_5.q_s_32.ep_200--use_schema--label_num_schema2/"
    elif args.dataset == "na":
        if args.shot == 1:
            path = "../data/stanford/stanford.1.spt_s_1.q_s_32.ep_200--use_schema--label_num_schema2/"
        else:
            path = "../data/stanford/stanford.1.spt_s_5.q_s_32.ep_200--use_schema--label_num_schema2/"
    elif args.dataset == "we":
        if args.shot == 1:
            path = "../data/stanford/stanford.2.spt_s_1.q_s_32.ep_200--use_schema--label_num_schema2/"
        else:
            path = "../data/stanford/stanford.2.spt_s_5.q_s_32.ep_200--use_schema--label_num_schema2/"
    elif args.dataset == "cwoz_at":
        if args.shot == 1:
            path = "../data/crosswoz/crosswoz.0.spt_s_1.q_s_32.ep_200/"
        else:
            path = "../data/crosswoz/crosswoz.0.spt_s_5.q_s_32.ep_200/"
    elif args.dataset == "cwoz_ho":
        if args.shot == 1:
            path = "../data/crosswoz/crosswoz.1.spt_s_1.q_s_32.ep_200/"
        else:
            path = "../data/crosswoz/crosswoz.1.spt_s_5.q_s_32.ep_200/"
    elif args.dataset == "cwoz_re":
        if args.shot == 1:
            path = "../data/crosswoz/crosswoz.2.spt_s_1.q_s_32.ep_200/"
        else:
            path = "../data/crosswoz/crosswoz.2.spt_s_5.q_s_32.ep_200/"
    elif args.dataset == "jwoz_sh":
        if args.shot == 1:
            path = "../data/jmultiwoz/jmultiwoz.0.spt_s_1.q_s_32.ep_200/"
        else:
            path = "../data/jmultiwoz/jmultiwoz.0.spt_s_5.q_s_32.ep_200/"
    elif args.dataset == "jwoz_ho":
        if args.shot == 1:
            path = "../data/jmultiwoz/jmultiwoz.1.spt_s_1.q_s_32.ep_200/"
        else:
            path = "../data/jmultiwoz/jmultiwoz.1.spt_s_5.q_s_32.ep_200/"
    elif args.dataset == "jwoz_re":
        if args.shot == 1:
            path = "../data/jmultiwoz/jmultiwoz.2.spt_s_1.q_s_32.ep_200/"
        else:
            path = "../data/jmultiwoz/jmultiwoz.2.spt_s_5.q_s_32.ep_200/"
    else:
        raise ValueError(
            'args.dataset should be one of'
            '[it, ac, at, fo, tr, sh, sc, na, we, cwoz_at, cwoz_ho, cwoz_re, jwoz_sh, jwoz_ho, jwoz_re]')

    if args.dataset in ['it', 'ac', 'at', 'fo', 'tr', 'sh']:
        label_dict = _load_toursg_label()
    elif args.dataset in ['cwoz_at', 'cwoz_ho', 'cwoz_re']:
        label_dict = _load_cwoz_label()
    elif args.dataset in ['jwoz_sh', 'jwoz_ho', 'jwoz_re']:
        label_dict = _load_jwoz_label()
    else:
        label_dict = _load_stanford_label()

    label = {}

    # loader training data
    with open(path + "train.json", "r", encoding="utf-8") as f:
        train_data = []
        text_len = []
        single = 0
        multi = 0
        data_raw = json.load(f)
        for key, items in data_raw.items():
            for line in items[0]:
                labels = line["support"]["labels"] + line["query"]["labels"]
                for l in labels:
                    if len(l) > 1:
                        multi += 1
                    else:
                        single += 1

                    for l1 in l:
                        if l1 not in label:
                            label[l1] = 1
                        else:
                            label[l1] += 1

                contents = line["support"]["seq_ins"] + line["query"]["seq_ins"]
                for cont in contents:
                    text_len.append(len(cont))

                support_label = []
                label_num = []
                for v in line["support"]["labels"]:
                    temp = []
                    for v1 in v:
                        temp.append(label_dict[v1])
                    support_label.append(temp)
                    label_num.append(len(v))

                train_data.append({
                    "support": {"seq_ins": line["support"]["seq_ins"], "label_seq": support_label,
                                "labels": line["support"]["labels"], "label_num": label_num},
                    "query": {"seq_ins": line["query"]["seq_ins"], "labels": line["query"]["labels"]},
                    "is_train": True
                })

        tprint("train: sample with single label: {}, sample with multi-label: {}".format(single, multi))
        train_num = len(text_len)
        tprint('Training data avg len: {}'.format(sum(text_len) / train_num))

    # loader dev data
    with open(path + "dev.json", "r", encoding="utf-8") as f:
        dev_data = []
        text_len = []
        single = 0
        multi = 0
        data_raw = json.load(f)
        for key, items in data_raw.items():
            for line in items[0]:

                contents = line["support"]["seq_ins"] + line["query"]["seq_ins"]
                for cont in contents:
                    text_len.append(len(cont))

                labels = line["support"]["labels"] + line["query"]["labels"]
                for l in labels:
                    if len(l) > 1:
                        multi += 1
                    else:
                        single += 1

                    for l1 in l:
                        if l1 not in label:
                            label[l1] = 1
                        else:
                            label[l1] += 1

                support_label = []
                label_num = []
                for v in line["support"]["labels"]:
                    temp = []
                    for v1 in v:
                        temp.append(label_dict[v1])
                    support_label.append(temp)
                    label_num.append(len(v))

                dev_data.append({
                    "support": {"seq_ins": line["support"]["seq_ins"], "label_seq": support_label,
                                "labels": line["support"]["labels"], "label_num": label_num},
                    "query": {"seq_ins": line["query"]["seq_ins"], "labels": line["query"]["labels"]},
                    "is_train": False
                })

        dev_num = len(text_len)
        tprint("dev: sample with single label: {}, sample with multi-label: {}".format(single, multi))
        tprint('dev data avg len: {}'.format(sum(text_len) / dev_num))

    # loader test data
    with open(path + "test.json", "r", encoding="utf-8") as f:
        test_data = []
        text_len = []
        single = 0
        multi = 0
        data_raw = json.load(f)
        for key, items in data_raw.items():
            for line in items[0]:

                contents = line["support"]["seq_ins"] + line["query"]["seq_ins"]
                for cont in contents:
                    text_len.append(len(cont))

                labels = line["support"]["labels"] + line["query"]["labels"]
                for l in labels:
                    if len(l) > 1:
                        multi += 1
                    else:
                        single += 1

                    for l1 in l:
                        if l1 not in label:
                            label[l1] = 1
                        else:
                            label[l1] += 1

                support_label = []
                label_num = []
                for v in line["support"]["labels"]:
                    temp = []
                    for v1 in v:
                        temp.append(label_dict[v1])
                    support_label.append(temp)
                    label_num.append(len(v))

                test_data.append({
                    "support": {"seq_ins": line["support"]["seq_ins"], "label_seq": support_label,
                                "labels": line["support"]["labels"], "label_num": label_num},
                    "query": {"seq_ins": line["query"]["seq_ins"], "labels": line["query"]["labels"]},
                    "is_train": False
                })

        tprint("test: sample with single label: {}, sample with multi-label: {}".format(single, multi))
        test_num = len(text_len)
        tprint('test data avg len: {}'.format(sum(text_len) / test_num))

    tprint('Class balance:')
    print(label, len(label))
    tprint('#train {}, #val {}, #test {}'.format(train_num, dev_num, test_num))
    return train_data, dev_data, test_data

def _build_label_full(args, s_labels, q_labels):
    if args.dataset in ['it', 'ac', 'at', 'fo', 'tr', 'sh']:
        label_dict = {'POSITIVE': 0, 'WHO': 1, 'INFO': 2, 'WHAT': 3, 'HOW_MUCH': 4, 'RECOMMEND': 5, 'CONFIRM': 6,
                      'WHICH': 7, 'THANK': 8, 'COMMIT': 9, 'NEGATIVE': 10, 'EXPLAIN': 11, 'ACK': 12, 'PREFERENCE': 13,
                      'WHEN': 14, 'HOW_TO': 15, 'WHERE': 16, 'ENOUGH': 17, 'OPENING': 18}
    elif args.dataset in ['cwoz_at', 'cwoz_ho', 'cwoz_re']:
        label_dict = {'推荐景点': 0, '询问电话': 1, '询问地址': 2, '询问时间': 3, '询问周边': 4,
                      '询问票价': 5, '询问评分': 6, '问好': 7, '询问服务': 8, '询问设施': 9, '推荐酒店': 10,
                      '询问类型': 11, '询问价格': 12, '推荐餐馆': 13, '推荐菜': 14, '询问人均消费': 15}
    elif args.dataset in ['jwoz_sh', 'jwoz_ho', 'jwoz_re']:
        label_dict = {'宿泊を探': 0, '時間': 1, '電話': 2, '住所を/エリア': 3, 'サービス': 4, 'り駅': 5, '予算/値段': 6,
                      'わかりました/はい': 7,
                      '予約': 8, 'ありがとうござい': 9, 'レストラン/飲食': 10, 'ショッピング': 11}
    else:
        label_dict = {'request_low_temperature': 0, 'request_time': 1, 'appreciate': 2, 'request_temperature': 3,
                      'request_weather': 4, 'inform': 5, 'request_high_temperature': 6, 'query': 7, 'request_poi': 8,
                      'request_traffic': 9, 'request_address': 10, 'request_route': 11, 'confirm': 12,
                      'show_in_screen': 13,
                      'navigate': 14, 'schedule': 15, 'command_appointment': 16, 'remind': 17,
                      'request_information': 18,
                      'list_schedule': 19, 'request_party': 20, 'request_agenda': 21, 'request_location': 22,
                      'request_date': 23}

    max_len = len(label_dict)
    id2label = {}
    for key, item in label_dict.items():
        id2label[item] = key

    s_label_number = np.zeros([len(s_labels), max_len], dtype=np.int64)
    q_label_number = np.zeros([len(q_labels), max_len], dtype=np.int64)

    for i, l in enumerate(s_labels):
        for s in l:
            s_label_number[i, label_dict[s]] = 1

    for i, l in enumerate(q_labels):
        for s in l:
            q_label_number[i, label_dict[s]] = 1

    return s_label_number, q_label_number, id2label


def _data_label_knowledge(tokenizer, args):
    if args.dataset in ['it', 'ac', 'at', 'fo', 'tr', 'sh']:
        labels = pd.read_csv("../data/toursg_label_content.csv")
    elif args.dataset in ['cwoz_at', 'cwoz_ho', 'cwoz_re']:
        labels = pd.read_csv("../data/cwoz_label_content.csv")
    elif args.dataset in ['jwoz_sh', 'jwoz_ho', 'jwoz_re']:
        labels = pd.read_csv("../data/jwoz_label_content.csv")
    else:
        labels = pd.read_csv("../data/stanford_label_content.csv")

    contents = []
    masks = []
    for i, line in enumerate(labels["content"]):
        sent = line + " [SEP] " + labels["label"][i]
        tokens = tokenizer(sent, return_tensors="pt")
        input_ids, attn_mask = list(map(str, tokens['input_ids'][0].tolist())), list(
            map(str, tokens['attention_mask'][0].tolist()))
        contents.append(input_ids)
        masks.append(attn_mask)

    lens = np.array([len(e) for e in contents])
    max_text_len = max(lens)

    text = np.zeros([len(labels["content"]), max_text_len], dtype=np.int64)
    text_mask = np.zeros([len(labels["content"]), max_text_len], dtype=np.int64)
    for i in range(len(labels["content"])):
        text[i, :len(list(map(int, contents[i])))] = np.array(list(map(int, contents[i])))
        text_mask[i, :len(list(map(int, masks[i])))] = np.array(list(map(int, masks[i])))

    return {"text": torch.LongTensor(text).cuda(args.cuda),
            "attn_mask": torch.LongTensor(text_mask).cuda(args.cuda)}

def _token_level(args, tokens, tokenizer, mlm_bert):
    mlm_bert.cuda(args.cuda)
    tokens['input_ids'], _ = masked.mask_tokens(inputs=tokens['input_ids'].cpu(), tokenizer=tokenizer, mlm_probability=0.25)
    tokens['input_ids'] = tokens['input_ids'].to(mlm_bert.device)
    tokens['attention_mask'] = tokens['attention_mask'].to(mlm_bert.device)
    tokens['token_type_ids'] = tokens['token_type_ids'].to(mlm_bert.device)
    with torch.no_grad():
        outputs = mlm_bert(**tokens)
        logits = outputs.logits.detach().cpu()
    tokens['input_ids'] = tokens['input_ids'].detach().cpu()
    tokens['attention_mask'] = tokens['attention_mask'].detach().cpu()
    tokens['token_type_ids'] = tokens['token_type_ids'].detach().cpu()
    mask_index = torch.where(tokens['input_ids'] == tokenizer.mask_token_id)[1].detach().cpu().tolist()
    predicted_ids = torch.argmax(logits[0, mask_index], dim=-1).detach().cpu().tolist()
    for i, token in enumerate(predicted_ids):
        tokens['input_ids'][0][mask_index[i]] = token
    return tokens

def _data_to_nparray(data, args):
    '''
        Convert the data into a dictionary of np arrays for speed.
    '''

    tokenizer = BertTokenizer.from_pretrained(args.pretrained)
    mlm_bert = BertForMaskedLM.from_pretrained(args.pretrained)

    label_kno = _data_label_knowledge(tokenizer, args)

    all_episodes = []
    for episode in data:
        # for support
        contents = []
        masks = []
        for i, text in enumerate(episode["support"]["seq_ins"]):
            if args.use_label:
                text = ' '.join(text + [" [SEP] "] + [" [SEP] ".join(episode["support"]["label_seq"][i])])
            else:
                text = ' '.join(text)
            tokens = tokenizer(text, return_tensors="pt")
            input_ids, attn_mask = list(map(str, tokens['input_ids'][0].tolist())), list(
                map(str, tokens['attention_mask'][0].tolist()))
            contents.append(input_ids)
            masks.append(attn_mask)

        lens = np.array([len(e) for e in contents])
        max_text_len = max(lens)

        s_text = np.zeros([len(episode["support"]["seq_ins"]), max_text_len], dtype=np.int64)
        s_text_mask = np.zeros([len(episode["support"]["seq_ins"]), max_text_len], dtype=np.int64)
        for i in range(len(episode["support"]["seq_ins"])):
            s_text[i, :len(list(map(int, contents[i])))] = np.array(list(map(int, contents[i])))
            s_text_mask[i, :len(list(map(int, masks[i])))] = np.array(list(map(int, masks[i])))

        # for query
        contents = []
        masks = []
        for text in episode["query"]["seq_ins"]:
            tokens = tokenizer(' '.join(text), return_tensors="pt")
            input_ids, attn_mask = list(map(str, tokens['input_ids'][0].tolist())), list(
                map(str, tokens['attention_mask'][0].tolist()))
            contents.append(input_ids)
            masks.append(attn_mask)

        lens = np.array([len(e) for e in contents])
        max_text_len = max(lens)

        q_text = np.zeros([len(episode["query"]["seq_ins"]), max_text_len], dtype=np.int64)
        q_text_mask = np.zeros([len(episode["query"]["seq_ins"]), max_text_len], dtype=np.int64)
        for i in range(len(episode["query"]["seq_ins"])):
            q_text[i, :len(list(map(int, contents[i])))] = np.array(list(map(int, contents[i])))
            q_text_mask[i, :len(list(map(int, masks[i])))] = np.array(list(map(int, masks[i])))

        s_label, q_label, id2label = _build_label_full(args, episode["support"]["labels"], episode["query"]["labels"])

        if args.cuda != -1:
            all_episodes.append({
                "support": {"text": torch.LongTensor(s_text).cuda(args.cuda),
                            "attn_mask": torch.LongTensor(s_text_mask).cuda(args.cuda),
                            "s_label": torch.LongTensor(s_label).cuda(args.cuda)},
                "query": {"text": torch.LongTensor(q_text).cuda(args.cuda),
                          "attn_mask": torch.LongTensor(q_text_mask).cuda(args.cuda),
                          "q_label": torch.LongTensor(q_label).cuda(args.cuda)},
                "label": label_kno,
                "id2label": id2label
            })
        else:
            all_episodes.append({
                "support": {"text": torch.LongTensor(s_text),
                            "attn_mask": torch.LongTensor(s_text_mask),
                            "s_label": torch.LongTensor(s_label)},
                "query": {"text": torch.LongTensor(q_text),
                          "attn_mask": torch.LongTensor(q_text_mask),
                          "q_label": torch.LongTensor(q_label)},
                "label": label_kno,
                "id2label": id2label
            })

    return all_episodes


def _dckpn_label_seq(s_labels, q_labels, args):
    if args.dataset in ['it', 'ac', 'at', 'fo', 'tr', 'sh']:
        label_dict = {'POSITIVE': 0, 'WHO': 1, 'INFO': 2, 'WHAT': 3, 'HOW_MUCH': 4, 'RECOMMEND': 5, 'CONFIRM': 6,
                      'WHICH': 7, 'THANK': 8, 'COMMIT': 9, 'NEGATIVE': 10, 'EXPLAIN': 11, 'ACK': 12, 'PREFERENCE': 13,
                      'WHEN': 14, 'HOW_TO': 15, 'WHERE': 16, 'ENOUGH': 17, 'OPENING': 18}
    elif args.dataset in ['cwoz_at', 'cwoz_ho', 'cwoz_re']:
        label_dict = {'推荐景点': 0, '询问电话': 1, '询问地址': 2, '询问时间': 3, '询问周边': 4,
                      '询问票价': 5, '询问评分': 6, '问好': 7, '询问服务': 8, '询问设施': 9, '推荐酒店': 10,
                      '询问类型': 11, '询问价格': 12, '推荐餐馆': 13, '推荐菜': 14, '询问人均消费': 15}
    elif args.dataset in ['jwoz_sh', 'jwoz_ho', 'jwoz_re']:
        label_dict = {'宿泊を探': 0, '時間': 1, '電話': 2, '住所を/エリア': 3, 'サービス': 4, 'り駅': 5, '予算/値段': 6,
                      'わかりました/はい': 7, '予約': 8, 'ありがとうござい': 9, 'レストラン/飲食': 10, 'ショッピング': 11}
    else:
        label_dict = {'request_low_temperature': 0, 'request_time': 1, 'appreciate': 2, 'request_temperature': 3,
                      'request_weather': 4, 'inform': 5, 'request_high_temperature': 6, 'query': 7, 'request_poi': 8,
                      'request_traffic': 9, 'request_address': 10, 'request_route': 11, 'confirm': 12,
                      'show_in_screen': 13,
                      'navigate': 14, 'schedule': 15, 'command_appointment': 16, 'remind': 17,
                      'request_information': 18,
                      'list_schedule': 19, 'request_party': 20, 'request_agenda': 21, 'request_location': 22,
                      'request_date': 23}

    label_name = []
    for l in s_labels:
        for l1 in l:
            label_name.append(l1)

    max_len = len(label_dict)
    id2label = {}
    for key, item in label_dict.items():
        id2label[item] = key

    s_label_number = np.zeros([len(label_name), max_len], dtype=np.int64)
    q_label_number = np.zeros([len(q_labels), max_len], dtype=np.int64)

    for i, l in enumerate(label_name):
        s_label_number[i, label_dict[l]] = 1

    for i, l in enumerate(q_labels):
        for s in l:
            q_label_number[i, label_dict[s]] = 1

    return s_label_number, q_label_number, id2label


def _data_dckpn(data, args, flag=None):
    '''
        Convert the data into a dictionary of np arrays for speed.
    '''

    tokenizer = BertTokenizer.from_pretrained(args.pretrained)
    if flag == "test":
        tprint("Perform a token-level attack")
        mlm_bert = BertForMaskedLM.from_pretrained(args.pretrained)

    all_episodes = []
    for episode in data:
        # for support
        contents = []
        masks = []
        label_num = []
        label_name = []
        for i, text in enumerate(episode["support"]["seq_ins"]):
            for label in episode["support"]["label_seq"][i]:
                label_num.append(episode["support"]["label_num"][i])
                label_name.append(label)
                text = ' '.join(text) + " [SEP] " + label
                tokens = tokenizer(text, return_tensors="pt")
                input_ids, attn_mask = list(map(str, tokens['input_ids'][0].tolist())), list(
                    map(str, tokens['attention_mask'][0].tolist()))
                contents.append(input_ids)
                masks.append(attn_mask)

        lens = np.array([len(e) for e in contents])
        max_text_len = max(lens)

        s_text = np.zeros([len(contents), max_text_len], dtype=np.int64)
        s_text_mask = np.zeros([len(contents), max_text_len], dtype=np.int64)
        for i in range(len(contents)):
            s_text[i, :len(list(map(int, contents[i])))] = np.array(list(map(int, contents[i])))
            s_text_mask[i, :len(list(map(int, masks[i])))] = np.array(list(map(int, masks[i])))

        # for query
        contents = []
        masks = []
        for text in episode["query"]["seq_ins"]:
            tokens = tokenizer(' '.join(text), return_tensors="pt")
            if flag == "test":
                tokens = _token_level(args, tokens, tokenizer, mlm_bert)
            input_ids, attn_mask = list(map(str, tokens['input_ids'][0].tolist())), list(
                map(str, tokens['attention_mask'][0].tolist()))
            contents.append(input_ids)
            masks.append(attn_mask)

        lens = np.array([len(e) for e in contents])
        max_text_len = max(lens)
        q_length = len(contents)

        q_text = np.zeros([len(episode["query"]["seq_ins"]), max_text_len], dtype=np.int64)
        q_text_mask = np.zeros([len(episode["query"]["seq_ins"]), max_text_len], dtype=np.int64)
        for i in range(len(episode["query"]["seq_ins"])):
            q_text[i, :len(list(map(int, contents[i])))] = np.array(list(map(int, contents[i])))
            q_text_mask[i, :len(list(map(int, masks[i])))] = np.array(list(map(int, masks[i])))

        # for label
        label_name = label_name + q_length * ["none"]
        contents = []
        masks = []
        for text in label_name:
            tokens = tokenizer(text, return_tensors="pt")
            input_ids, attn_mask = list(map(str, tokens['input_ids'][0].tolist())), list(
                map(str, tokens['attention_mask'][0].tolist()))
            contents.append(input_ids)
            masks.append(attn_mask)

        lens = np.array([len(e) for e in contents])
        max_text_len = max(lens)

        l_text = np.zeros([len(label_name), max_text_len], dtype=np.int64)
        l_text_mask = np.zeros([len(label_name), max_text_len], dtype=np.int64)
        for i in range(len(label_name)):
            l_text[i, :len(list(map(int, contents[i])))] = np.array(list(map(int, contents[i])))
            l_text_mask[i, :len(list(map(int, masks[i])))] = np.array(list(map(int, masks[i])))

        s_label, q_label, id2label = _dckpn_label_seq(episode["support"]["labels"], episode["query"]["labels"], args)

        if args.cuda != -1:
            all_episodes.append({
                "support": {"text": torch.LongTensor(s_text).cuda(args.cuda),
                            "attn_mask": torch.LongTensor(s_text_mask).cuda(args.cuda),
                            "s_label": torch.LongTensor(s_label).cuda(args.cuda),
                            "label_num": torch.LongTensor(label_num).cuda(args.cuda)},
                "query": {"text": torch.LongTensor(q_text).cuda(args.cuda),
                          "attn_mask": torch.LongTensor(q_text_mask).cuda(args.cuda),
                          "q_label": torch.LongTensor(q_label).cuda(args.cuda)},
                "label": {"text": torch.LongTensor(l_text).cuda(args.cuda),
                          "attn_mask": torch.LongTensor(l_text_mask).cuda(args.cuda)},
                "id2label": id2label
            })
        else:
            all_episodes.append({
                "support": {"text": torch.LongTensor(s_text),
                            "attn_mask": torch.LongTensor(s_text_mask),
                            "s_label": torch.LongTensor(s_label),
                            "label_num": torch.LongTensor(label_num)},
                "query": {"text": torch.LongTensor(q_text),
                          "attn_mask": torch.LongTensor(q_text_mask),
                          "q_label": torch.LongTensor(q_label)},
                "label": {"text": torch.LongTensor(l_text).cuda(args.cuda),
                          "attn_mask": torch.LongTensor(l_text_mask).cuda(args.cuda)},
                "id2label": id2label
            })

    return all_episodes


def load_dataset(args):
    tprint('Loading data: {}, shot: {}'.format(args.dataset, args.shot))

    train_data, dev_data, test_data = _load_json(args)

    # Convert everything into np array for fast data loading
    if args.classifier in ['dckpn', 'mmn']:
        train_data = _data_dckpn(train_data, args)
        val_data = _data_dckpn(dev_data, args)
        test_data = _data_dckpn(test_data, args)
    else:
        train_data = _data_to_nparray(train_data, args)
        val_data = _data_to_nparray(dev_data, args)
        test_data = _data_to_nparray(test_data, args)

    return train_data, val_data, test_data
