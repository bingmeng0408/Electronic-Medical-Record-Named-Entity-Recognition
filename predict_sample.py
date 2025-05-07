from Bert_ner.Bert_module import *

# 初始化模型
conf = Config()
model = BERT_CRF(conf.dropout, conf.tag2id)
# 如果无显卡，用cpu运行，这里参数改为：map_location='cpu'
# model.load_state_dict(torch.load(conf.save_path, map_location=conf.device))
model.load_state_dict(torch.load(conf.save_path, map_location='cpu'),strict=False)
model = model.to(conf.device)
def predict_entities(sample):
    sample1 = []
    for word in sample:
        sample1.append(word)
    model.eval()
    data = tokenizer.encode_plus(sample1, add_special_tokens=False)
    text = data['input_ids']
    mask = data['attention_mask']
    x = torch.tensor(text, dtype=torch.long, device=conf.device).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long, device=conf.device).unsqueeze(0)
    result = model(x, mask)[0]
    id2tag = {idx: tag for tag, idx in conf.tag2id.items()}
    predict_list = [id2tag.get(i) for i in result]
    entities = []
    current_entity = None
    current_start = 0
    for i, (token, tag) in enumerate(zip(sample1, predict_list)):
        if tag == 'O':
            if current_entity is not None:
                entity_text = ''.join(sample1[current_start:i])
                entities.append([entity_text, current_entity])
                current_entity = None
        elif '-B' in tag:
            entity_type = tag.split('-')[0]
            if current_entity is not None:
                entity_text = ''.join(sample1[current_start:i])
                entities.append([entity_text, current_entity])
            current_entity = entity_type
            current_start = i
        elif '-I' in tag:
            entity_type = tag.split('-')[0]
            if current_entity is None:
                current_entity = entity_type
                current_start = i
            elif entity_type != current_entity:
                entity_text = ''.join(sample1[current_start:i])
                entities.append([entity_text, current_entity])
                current_entity = entity_type
                current_start = i
    if current_entity is not None:
        entity_text = ''.join(sample1[current_start:])
        entities.append([entity_text, current_entity])
    converted=[]
    for item in entities:
        eng_type = item[1].upper() if isinstance(item[1], str) else item[1]
        eng2chine = {v: k for k, v in conf.labels.items()}
        if eng_type in eng2chine:
            converted.append([item[0], eng2chine[eng_type]])
        else:
            converted.append(item)
    return converted,predict_list,sample1,sample


if __name__ == '__main__':
    # 患者精神状况好，无发热，诉右髋部疼痛，饮食差，二便正常，查体：神清，各项生命体征平稳，心肺腹查体未见异常。右髋部压痛，右下肢皮牵引固定好，无松动，右足背动脉搏动好，足趾感觉运动正常。
    # 小明父亲患有冠心病及糖尿病，无手术外伤史及药物过敏史
    # 老年男性，主因：恶心、呕吐1天。于2016-04-16,16:06入院。
    converted,predict_list,sample1,sample = predict_entities(
        sample="老年男性，右侧股骨粗隆间骨折。于2016-04-16,16:06入院。")
    print(sample)
    print(sample1)
    print(predict_list)
    print(converted)
