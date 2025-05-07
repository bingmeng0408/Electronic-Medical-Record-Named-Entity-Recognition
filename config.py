import os
import torch
import json

cur = os.getcwd()
class Config(object):
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
        # 苹果M1芯片及其以上的电脑（使用GPU）
        # self.device = 'mps'
        self.origin_train_path = os.path.join(cur, 'data_origin')
        self.train_path = os.path.join(cur, 'data\\train.txt')
        self.bert_path = os.path.join(cur, 'bert-base-chinese')
        self.save_path = os.path.join(cur, 'save_model\\bert_crf_best.pth')
        self.tag2id = json.load(open(os.path.join(cur, 'data\\tag2id.json'), 'r', encoding='utf-8'))
        self.labels = json.load(open(os.path.join(cur, 'data\\labels.json'), 'r', encoding='utf-8'))
        self.epochs = 10
        self.batch_size = 8
        self.lr = 1e-6
        self.dropout = 0.1

if __name__ == '__main__':
    conf = Config()
    print(conf.train_path)
    # print(conf.tag2id)