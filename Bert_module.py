import torch
import torch.nn as nn
from TorchCRF import CRF
from config import *

from Bert_ner.dataloader import *
from transformers import BertModel
conf = Config()
class BERT_CRF(nn.Module):
    def __init__(self, dropout, tag2id):
        super(BERT_CRF, self).__init__()
        self.tag_to_ix = tag2id
        self.tag_size = len(tag2id)
        # BERT 模型
        self.bert = BertModel.from_pretrained(conf.bert_path)
        self.bert_dim = 768  # BERT 输出维度
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Linear layer
        self.hidden2tag = nn.Linear(self.bert_dim, self.tag_size)
        # CRF
        self.crf = CRF(self.tag_size)

    def forward(self, x, mask):
        # BERT 输出
        bert_output = self.bert(x, attention_mask=mask)
        bert_embeddings = bert_output.last_hidden_state
        outputs = self.dropout(bert_embeddings)
        # Linear
        outputs = self.hidden2tag(outputs)
        outputs = outputs * mask.unsqueeze(-1)
        # CRF 解码
        outputs = self.crf.viterbi_decode(outputs, mask)
        return outputs
    # CRF 损失
    def log_likelihood(self, x, tags, mask):
        # BERT 输出
        bert_output = self.bert(x, attention_mask=mask)
        bert_embeddings = bert_output.last_hidden_state
        outputs = self.dropout(bert_embeddings)
        # Linear
        outputs = self.hidden2tag(outputs)
        outputs = outputs * mask.unsqueeze(-1)

        # CRF 损失
        return -self.crf(outputs, tags, mask)


if __name__ == '__main__':
    conf = Config()
    model = BERT_CRF(
        dropout=conf.dropout,
        tag2id=conf.tag2id
    )
    # print(model)

    train_dataloader, dev_dataloader = get_data()
    for x, y, mask in train_dataloader:
        mask = mask.to(torch.bool)
        loss = model.log_likelihood(x, y, mask)
        print(loss)
        break