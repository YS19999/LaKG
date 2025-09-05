import json
import numpy as np
import torch

vocab = []
with open("train.json", "r", encoding="utf-8") as f:
    data1 = json.load(f)
    for key, item in data1.items():
        print(key)
        for raw in item[0]:
            vocab.append(sum(raw["support"]["seq_ins"], []))

with open("dev.json", "r", encoding="utf-8") as f:
    data1 = json.load(f)
    for key, item in data1.items():
        print(key)
        for raw in item[0]:
            vocab.append(sum(raw["support"]["seq_ins"], []))

with open("test.json", "r", encoding="utf-8") as f:
    data1 = json.load(f)
    for key, item in data1.items():
        print(key)
        for raw in item[0]:
            vocab.append(sum(raw["support"]["seq_ins"], []))

vocab = list(set(sum(vocab, [])))
print(vocab)
print(len(vocab))
with open("../vocab_it5.json", "w", encoding="utf-8") as file:

    dicts = [{"vocab": vocab}]
    json.dump(dicts, file)

