import json
import os
from config import *
conf=Config()
class TransferData:
    def __init__(self):
        self.label_dict = conf.labels
        self.cate_dict =conf.tag2id
        self.origin_path = conf.origin_train_path
        self.train_filepath = conf.train_path
    def transfer(self):
        with open(self.train_filepath, 'w', encoding='utf-8') as f:
            for root,dirs,files in os.walk(self.origin_path):
                for file in files:
                    filepath = os.path.join(root, file)
                    if 'original' not in filepath:
                        continue
                    label_filepath = filepath.replace('.txtoriginal','')
                    print(filepath, '\t\t', label_filepath)

                    res_dict = {}
                    for line in open(label_filepath, 'r', encoding='utf-8'):
                        res = line.strip().split('	')
                        start = int(res[1])
                        end = int(res[2])
                        label = res[3]
                        label_id = self.label_dict.get(label)
                        for i in range(start, end+1):
                            if i == start:
                                label_cate = label_id + '-B'
                            else:
                                label_cate = label_id + '-I'
                            res_dict[i] = label_cate
                    with open(filepath, 'r', encoding='utf-8') as fr:
                        content = fr.read().strip()
                        for indx, char in enumerate(content):
                            if char==' ':
                                continue
                            char_label = res_dict.get(indx, 'O')
                            f.write(char + '\t' + char_label + '\n')





if __name__ == '__main__':
    handler = TransferData()
    handler.transfer()