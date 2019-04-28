# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import logging
import torch
import torch.nn as nn
import torch.utils.data
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
#
# print(os.listdir("../input"))
#
# # Any results you write to the current directory are saved as output.
#
# os.system('pip install --no-index --find-links="../input/pytorchpretrainedbert/" pytorch_pretrained_bert')
# # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows


class TrainEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row = self.data.iloc[item]

        text = "[CLS] {} [SEP".format(row['comment_text'])
        tokens = tokenizer.tokenize(text)
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        input_id = torch.tensor(input_id)

        segment_id = [0] * len(input_id)
        segment_id = torch.tensor(segment_id)

        input_mask = [1] * len(input_id)
        input_mask = torch.tensor(input_mask)

        label_id = row['target']
        label_id = (label_id > 0.5).astype(np.int64)
        label_id = torch.tensor(label_id)

        return input_id, segment_id, input_mask, label_id


def pad(tensors):
    result = torch.zeros(len(tensors), max(t.size(0) for t in tensors), dtype=tensors[0].dtype)

    for i, t in enumerate(tensors):
        result[i, :t.size(0)] = tensors[i]

    return result


def collate_fn(batch):
    input_ids, segment_ids, input_mask, label_ids = zip(*batch)

    input_ids = pad(input_ids)
    segment_ids = pad(segment_ids)
    input_mask = pad(input_mask)
    label_ids = torch.tensor(label_ids)

    return input_ids, segment_ids, input_mask, label_ids


if __name__ == '__main__':
    # MODEL_PATH = '../input/torch-bert-weights/bert-base-uncased/bert-base-uncased'
    # TOKENIZER_PATH = '../input/bert-vocab/bert-base-uncased-vocab.txt'

    MODEL_PATH = 'bert-base-uncased'
    TOKENIZER_PATH = 'bert-base-uncased'

    logging.basicConfig(level=logging.INFO)

    # # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    #
    # # Tokenized input
    # text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    # tokenized_text = tokenizer.tokenize(text)
    #
    # # Mask a token that we will try to predict back with `BertForMaskedLM`
    # masked_index = 8
    # tokenized_text[masked_index] = '[MASK]'
    # assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet',
    #                           '##eer', '[SEP]']
    #
    # # Convert token to vocabulary indices
    # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    # segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    #
    # # Convert inputs to PyTorch tensors
    # tokens_tensor = torch.tensor([indexed_tokens])
    # segments_tensors = torch.tensor([segments_ids])

    NUM_LABELS = 2

    model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        cache_dir='./cache',
        num_labels=NUM_LABELS)
    for m in model.bert.parameters():
        m.requires_grad = False
    # self.dropout = nn.Dropout(config.hidden_dropout_prob)
    # self.classifier = nn.Linear(config.hidden_size, num_labels)

    train_eval_data = pd.read_csv('./data/toxic/train.csv')

    EPOCHS = 10
    BATCH_SIZE = 4  # FIXME:
    LR = 5e-5
    WARMUP = 0.1

    indices = np.random.permutation(len(train_eval_data))
    train_indices, eval_indices = indices[:-indices.shape[0] // 5], indices[-indices.shape[0] // 5:]

    train_dataset = TrainEvalDataset(train_eval_data.iloc[train_indices])
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collate_fn)

    # param_optimizer = list(model.named_parameters())
    param_optimizer = list(model.classifier.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=LR,
        warmup=WARMUP,
        t_total=EPOCHS * len(train_data_loader))

    for epoch in range(EPOCHS):
        model.train()
        for input_ids, segment_ids, input_mask, label_ids in tqdm(train_data_loader):
            logits = model(input_ids, segment_ids, input_mask, labels=None)

            loss = nn.CrossEntropyLoss()(logits.view(-1, NUM_LABELS), label_ids.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
