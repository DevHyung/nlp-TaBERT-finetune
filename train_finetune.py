from table_bert import Table, Column, TableBertModel
from transformers import *

from typing import List
from tqdm import tqdm
import random
import time
import json
import os
import argparse

import torch
from torch import nn, optim, from_numpy
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class table:
    def __init__(self, _qid, _rel, _id, _title,
                 _secTitle, _heading, _body, _caption, _numHeaderRows):
        self.qid: str = _qid
        self.rel: str = _rel
        self.id: str = _id
        self.title: str = _title
        self.secTitle: str = _secTitle
        self.heading: List[str] = _heading
        self.body: List[List[str]] = _body
        self.caption: str = _caption
        self.numHeaderRows: int = _numHeaderRows

    def __str__(self):  # for debug
        heading = "\t|||\t".join(self.heading)
        body = ''
        for row in self.body: body += "\t|||\t".join(row) + '\n'
        return (f"______________________________________\n"
                f"Rel:{self.rel}\n"
                f"Table ID:{self.id}\n"
                f"numHRows:{self.numHeaderRows}\n"
                f"Title : {self.title}\n"
                f"SecTitle : {self.secTitle}\n"
                f"Caption : {self.caption}\n\n"
                f"{heading}\n"
                f"_____\n"
                f"{body}")


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Later FIX 200930기록
        # : BERT나 TaBERT 빈껍데기를 불러오는것으로 바꿔보는것도
        self.Qmodel = BertModel.from_pretrained(BERT_MODEL)
        self.Tmodel = TableBertModel.from_pretrained(
            TABERT_MODEL_PATH,
        )

    def forward(self, q, tp, tn=None):
        # cosSim(queryCLS[1,768], (avgPoll(context)[1,768] + avgPool(column)[1,768]))
        cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
        avgPool = torch.nn.AdaptiveAvgPool2d([1, 768]).to(device)

        # query embedding
        inputs = {
            "input_ids": q['input_ids'].to(device),
            "attention_mask": q['attention_mask'].to(device),
            "token_type_ids": q['token_type_ids'].to(device)
        }
        qCLS = self.Qmodel(**inputs)[1].to(device)

        # table postive embedding
        table = Table(
            id=tp.title,
            header=[Column(h.strip(), 'text') for h in tp.heading],
            data=tp.body
        ).tokenize(self.Tmodel.tokenizer)

        context = tp.caption
        context_encoding, column_encoding, _ = self.Tmodel.encode(
            contexts=[self.Tmodel.tokenizer.tokenize(context)],
            tables=[table]
        )
        tp_concat_encoding = avgPool(context_encoding) + avgPool(column_encoding)
        q_tp_cos = cosine_similarity(qCLS, tp_concat_encoding)

        # if tn==None -> Eval시 사용
        if tn is not None:
            # Table negative
            table = Table(
                id=tn.title,
                header=[Column(d.strip(), 'text') for d in tn.heading],
                data=tn.body
            ).tokenize(self.Tmodel.tokenizer)

            context = tn.caption
            context_encoding, column_encoding, _ = self.Tmodel.encode(
                contexts=[self.Tmodel.tokenizer.tokenize(context)],
                tables=[table]
            )
            tn_concat_encoding = avgPool(context_encoding) + avgPool(column_encoding)
            q_tn_cos = cosine_similarity(qCLS, tn_concat_encoding)
        else: q_tn_cos = None
        return q_tp_cos, q_tn_cos


def MarginRankingLoss(input1, input2):
    y = torch.tensor(Y)
    margin = torch.tensor(MARGIN)
    # Later FIX 201005기록
    # : margin을 iteration 돌면서 증가하면서 하는것도 방법, 테크니컬튠할때 진행
    loss = Variable(torch.max(torch.zeros(1), -y*(input1-input2)+margin), requires_grad=True)
    return loss


def load_table_data(_filepath='./all.json')-> [List, List]:
    table_neg_list = []
    table_pos_list = []
    with open(_filepath, 'r') as f:
        lines = f.readlines()  # line numbers = 총 테이블 갯수
        for line in lines:
            if line.strip() == '': break
            # 테이블 기본 Meta data 파싱
            jsonStr = json.loads(line)
            tableId = jsonStr['docid']
            qid = jsonStr['qid']
            rel = jsonStr['rel']

            # Raw Json 파싱
            raw_json = json.loads(jsonStr['table']['raw_json'])
            title = raw_json['pgTitle']
            secTitle = raw_json['secondTitle']
            hRow = raw_json['numHeaderRows']
            row = raw_json['numDataRows']
            col = raw_json['numCols']
            caption = raw_json['caption']
            heading = raw_json['title']
            body = raw_json['data']

            if col == 0:  # Col & Row == 0인 열은 제외 -> Table 임베딩시 에러남
                if DEBUG: print('빈 열', col, tableId)
                continue
            elif row == 0:
                if DEBUG: print('빈 행', row, tableId)
                continue

            if str(rel) == '0':  # Pos==2,1 , Neg==0
                table_neg_list.append(table(qid, rel, tableId, title, secTitle,
                                          heading, body, caption, hRow))
            else:
                table_pos_list.append(table(qid, rel, tableId, title, secTitle,
                                          heading, body, caption, hRow))
    return table_pos_list, table_neg_list


def load_query_data(_filepath='./querys.txt')-> List:
    query_list = []
    with open(_filepath, 'r') as f:
        # id를 2자리 int로 만들어서 리스트에 넣음 -> qid로 정렬때문에
        lines = f.readlines()
        for l in lines:
            id, query = l.strip().split("\t")
            if len(id) == 1: id = "0" + id
            query_list.append(id + '\t' + query)
        # 중복제거 및 sorting
        query_list = list(set(query_list))
        query_list.sort()
        if DEBUG: print(query_list)
    return query_list


def build_dataset(_queryList, _tablePosList, _tableNegList):
    """
    :return: <Q, T_p, T_n> list
    """
    # Pre tokenize
    query_tensor_list = []
    for q in _queryList:
        input_ids = tokenizer.encode_plus(q,
                                          add_special_tokens=True,
                                          return_tensors='pt',
                                          truncation=True,
                                          padding=True,
                                          max_length=MAX_LENGTH)
        query_tensor_list.append(input_ids)

    # Build Triple data <Q, T_p, T_n>
    tp_list = []
    tn_list = []
    q_list = []
    for tp in tablePosList:
        for tn in tableNegList:
            if tp.qid == tn.qid:  # T_p 와 T_n의 QueryId를 같은것을 묶음
                query = query_tensor_list[int(tp.qid) - 1]  # QueryId가 1부터 시작이라 - 1
                q_list.append(query)
                tp_list.append(tp)
                tn_list.append(tn)
    return list(zip(q_list, tp_list, tn_list))


def get_now()-> str:
    now = time.localtime()
    return "%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)


def log(tag, text):
    if tag == 'i':   print("[INFO] " + text)
    elif tag == 'e': print("[ERROR] " + text)
    elif tag == 's': print("[SUCCESS] " + text)


if __name__ == "__main__":
    # Config Area START_______________
    # >   Model config
    TABERT_MODEL_PATH = './tabert_base_k3/model.bin'
    BERT_MODEL = 'bert-base-uncased'
    MAX_LENGTH = 42  # for BERT tokenizer max_length == 쿼리 평균 토크나이징 len
    SHUFFLE = True
    BATCH_SIZE = 64
    EPOCH_NUM = 20
    MARGIN = 1.0
    Y = 1.0
    LEARNING_RATE = 0.01
    device = torch.device("cuda:1")

    # >   Etc
    TABLE_JSON_FILE = './all.json'
    QUERY_TXT_FILE = "./querys.txt"
    DEBUG = False
    # Config Area END_________________

    # BERT Model Load
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    # Data Load
    tablePosList, tableNegList = load_table_data(_filepath=TABLE_JSON_FILE)
    queryList = load_query_data(_filepath=QUERY_TXT_FILE)
    log('s', "Load table, query datas")

    # Build Dataset <Q, T_p, T_n>
    trainDataset = build_dataset(queryList, tablePosList, tableNegList)
    log('i', f"Total Train Data set Cnt : {len(trainDataset)}")

    # Create model
    model = Model()
    model.to(device)
    log('s', "Create model")
    print(model)

    # Optimizer & Loss
    target = torch.ones(1).sign() # MarginRankingLoss Target
    criterion = nn.MarginRankingLoss(margin=MARGIN, reduction="mean")  # infer땐 빼라
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    for epoch in tqdm(range(EPOCH_NUM)):
        if SHUFFLE: random.shuffle(trainDataset)
        _epochLossList = []
        batchIdx = 1
        prevIdx = 0 # for batch idx
        for idx in range(BATCH_SIZE, len(trainDataset), BATCH_SIZE):
            _lossList = []
            for query, tableP, tableN in trainDataset[prevIdx:idx]:
                tpCos, tnCos = model(query, tableP, tableN)
                loss = criterion(torch.tensor([abs(torch.mean(tpCos))], requires_grad=True),
                                 torch.tensor([abs(torch.mean(tnCos))], requires_grad=True),
                                 target)
                _lossList.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'[{get_now()}] Epoch: {epoch + 1} | Batch: {batchIdx} | Loss: {sum(_lossList) / len(_lossList):.6f}')
            _epochLossList.append(sum(_lossList) / len(_lossList))
            batchIdx += 1
            prevIdx = idx
        print(f'\t[{get_now()}] Epoch: {epoch + 1} | Loss: {sum(_epochLossList) / len(_epochLossList):.6f}')
        torch.save(model.state_dict(), f"./epoch{epoch}_batch{BATCH_SIZE}_m{int(MARGIN)}_y{int(Y)}_lossformal.pt")



