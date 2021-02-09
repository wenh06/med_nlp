def show_case(results,ES):
    logs = []
    for res in results:
        log = []
        for v in res[:-1]:
            log.append((ES[str(v)][0],ES[str(v)][1]))
        log.append(['PLAN:'])
        for v in res[-1]:
            log[-1].append(ES[str(v)][0])
        logs.append(log)
    return logs

def ConvertDataFormat(data):
    data2 = {}
    data2['原文'] = data['原文']
    data2['实体识别结果'] = {}
    
    ann_entity = []
    last_start = -1
    last_ed = -1
    for e in data['实体识别结果']:
        word = e['word']
        tp = e['ner']
        start = e['beginPosition']
        ed = e['endPosition']
        if tp in ['P','CONJ']:
            continue
        #check cover
        if last_start<=start and last_ed>=ed:
            continue
        if start<=last_start and ed >= last_ed:
            ann_entity.pop()
        ann_entity.append([word,tp,start,ed])
        last_start = start
        last_ed = ed
    for i in range(len(ann_entity)):
        data2['实体识别结果'][str(i)] = ann_entity[i]
    return data2

SymbolNormTable = {f:t for f,t in zip(
     '，。！？【】（）％＃＠＆１２３４５６７８９０～-',
     ',.!?[]()%#@&1234567890~-')}

def ChineseTextNormalzation(text):
    text = text.replace('他','它')
    for c_c in SymbolNormTable:
        text = text.replace(c_c,SymbolNormTable[c_c])
    return text

def RecogEntity(data,centorid):
    entities = []
    for label in centorid['标注结果']:
        for es in label[:-1]:
            for v in es:
                entities.append(v)
    entities = list(set(entities))
    recognized_entity = []
    #print(entities)
    for eid in entities:
        flag = 1
        entity_i = centorid['实体识别结果'][str(eid)][0]
        entity_i = ChineseTextNormalzation(entity_i)
        for ejd in data['实体识别结果']:
            entity_j = data['实体识别结果'][ejd][0]
            entity_j = ChineseTextNormalzation(entity_j)
            if entity_i == entity_j:
                flag = 0
                break
        if flag:
            start = 0
            raw_text = ChineseTextNormalzation(data['原文'])
            while True:
                s = raw_text[start:].find(entity_i)
                if s == -1:
                    break
                start += s
                ed = start + len(entity_i)
                recognized_entity.append([data['原文'][start:ed],centorid['实体识别结果'][str(eid)][1],start,ed])
                start = ed
    
    if len(recognized_entity)>0:
        ann_entity = []
        for i in range(len(data['实体识别结果'])):
            ann_entity.append(data['实体识别结果'][str(i)])
        #print(len(ann_entity))
        for e in recognized_entity:
            word,tp,s1,e1 = e
            flag = 1
            for i in range(len(ann_entity)):
                if ann_entity[i][2]>s1:
                    flag = 0
                    break
            i += flag
            if i == len(ann_entity):
                ann_entity.append(e)
            else:
                #print(i)
                ann_entity = ann_entity[:i] + [e] + ann_entity[i:]
        #print(len(ann_entity))
        data['实体识别结果'] = {}
        for i in range(len(ann_entity)):
            data['实体识别结果'][str(i)] = ann_entity[i]

    return data
