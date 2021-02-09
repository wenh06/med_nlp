import numpy as np
import os
import json
import re

PLAN_TYPE = ['S_DOSAGE','UMETHOD','DURATION','FREQ','DOSAGE','UTIME']

def summarize_knowledge(centorid):
    cond_table = {}
    leaf_table = {}
    entity_table = {}
    
    for lid in range(len(centorid['标注结果'])):
        label = centorid['标注结果'][lid]
        #print(label)
        for ei in range(len(label)-2):
            for ej in range(ei+1,len(label)-1): 
                for e1 in label[ei]:
                    for e2 in label[ej]:
                        #print(ei,ej,e1,e2)
                        cond_table[str(e1)+'-'+str(e2)] = 1
        mx_e = -1
        for e in label[-2]:
            mx_e = max(mx_e,int(e))
        min_plan = 1000000000
        for e in label[-1]:
            min_plan = min(e,min_plan)
        if mx_e == min_plan -1:
            flag = 1
        else:
            flag = 2
        
        for e in label[-2]:
            leaf_table[str(e)] = flag
        
        for ei in range(len(label)-1):
            for e in label[ei]:
                word = centorid['实体识别结果'][str(e)]
                entity_table[e] = word
            
    return cond_table, leaf_table, entity_table
                

def MatchEntity(candidate_entity,centorid_entity_list):
    cw,ct,cs,ce = candidate_entity
    for eid in centorid_entity_list:
        word,tp,start,ed = centorid_entity_list[eid]
        if word == cw:
            return int(eid)
        if '肌酐清除率' in cw and '肌酐清除率' in word:
            if '30' in cw and '30' in word:
                return int(eid)
            if not ('30' in cw) and not ('30' in word):
                return int(eid)
    return -1



def EntityMatching(inx,sim_es,cen_es):
    e = sim_es[str(inx)]
    cand = []
    for i in range(len(cen_es)):
        i = str(i)
        if e[0] == cen_es[i][0]:
            cand.append(i)
    if len(cand) == 1:
        return cand[0]
    if len(cand) == 0:
        return -1
    look_back_length = 5
    sim_contexs = ''
    for k in range(inx+1,min(inx+1+look_back_length,len(sim_es))):
        sim_contexs += sim_es[str(k)][0]
    sim_contexs2 = {}
    sim_length = 0
    for v in sim_contexs:
        if not v in sim_contexs2:
            sim_contexs2[v] = 0
        sim_contexs2[v] += 1
        sim_length += 1

    cand_simlarity = []
    for cinx in cand:
        cinx = int(cinx)
        cen_contexts = ''
        for k in range(cinx+1,min(cinx+1+look_back_length,len(cen_es))):
            cen_contexts += cen_es[str(k)][0]
        #cen_contexts = set(cen_contexts)
        if len(sim_contexs) ==0:
            sim = 1-len(cen_contexts)
        else:
            cen_dict = {}
            for v in cen_contexts:
                if not v in cen_dict:
                    cen_dict[v] = 0
                cen_dict[v] += 1
            sim = 0
            for v in sim_contexs2:
                if v in cen_dict:
                    sim += min(cen_dict[v],sim_contexs2[v])
            
            sim = sim/(np.sqrt(sim_length)*np.sqrt(len(cen_contexts)))
        cand_simlarity.append(sim)
    cand_simlarity = np.array(cand_simlarity)

    return cand[cand_simlarity.argmax()]

def GetKnowledge1(data, centorid):
    cond_table, leaf_table, entity_table = summarize_knowledge(centorid)
    data_entity = data['实体识别结果']
    cen_entity = centorid['实体识别结果']
    knowledge_truples = []
    
    for i in range(len(data_entity)):
        wordi, tpi, _, _ = data_entity[str(i)]
        #inxi = MatchEntity(data_entity[str(i)],cen_entity)
        inxi = EntityMatching(i,data_entity,cen_entity)
        inxi = str(inxi)
        for j in range(i+1,len(data_entity)):
            #inxj = MatchEntity(data_entity[str(j)],cen_entity)
            inxj = EntityMatching(j,data_entity,cen_entity)
            inxj = str(inxj)
            wordj, tpj, _, _ = data_entity[str(j)]
            #print(str(inxi) + '-' + str(inxj))
            if str(inxi) + '-' + str(inxj) in cond_table:
                knowledge_truples.append([i,j,1])
                #print(i,j)
            if inxi in leaf_table and leaf_table[inxi] == 1 and tpj in PLAN_TYPE:
                flag_eq_tpi = 1
                flag = 1
                for k in range(i+1,j):
                    #print(len(data_entity))
                    #print(inxj)
                    tpk = data_entity[str(k)][1]
                    if flag_eq_tpi and tpk == tpi:
                        continue
                    else:
                        flag_eq_tpi = 0
                        if tpk in PLAN_TYPE:
                            continue
                        else:
                            flag = 0
                if flag:
                    knowledge_truples.append([i,j,2])

            elif inxi in leaf_table and leaf_table[inxi] == 2 and tpj in PLAN_TYPE:
                for k in range(i+1,j):
                    #print(len(data_entity))
                    #print(inxj)
                    tpk = data_entity[str(k)][1]
                    if  tpk in PLAN_TYPE:
                        break
                flag = 1
                for k2 in range(k+1,j):
                    tpk = data_entity[str(k2)][1]
                    if tpk in PLAN_TYPE:
                        continue
                    else:
                        flag = 0
                if flag:
                    knowledge_truples.append([i,j,2])

    return knowledge_truples

def get_indexed_es(entity):
    e2i = {}
    i2e = {}
    es = []
    
    counter ={}
    for i in range(len(entity)):
        word = entity[str(i)][0]
        if not word in counter:
            counter[word] = -1
        counter[word] += 1
        word = word + '-' + str(counter[word])
        e2i[word] = i
        i2e[i] = word
        es.append(word)
        
    
    return e2i, i2e, es


def get_error_list(centorid_kg,centorid):
    e2i, i2e, es = get_indexed_es(centorid['实体识别结果'])
    FLAG = {}
    for label in centorid['标注结果']:
        for ei in range(len(label)-1):
            for e1 in label[ei]:
                for e2 in label[ei+1]:
                    FLAG[str(e1)+'-'+str(e2)] = 1
    
    FLAG2 = {}
    for i,j,_ in centorid_kg:
        FLAG2[str(i)+'-'+str(j)] = 1
    
    Errors = {}
    for v in FLAG2:
        if not v in FLAG:
            Errors[v] = 0
    for v in FLAG:
        if not v in FLAG2:
            Errors[v] = 1
    
    return Errors


def GetKnowledge(data,centorid):
    
    centorid_kg = GetKnowledge1(centorid,centorid)
    data_kg = GetKnowledge1(data,centorid)
    
    centorid_e2i, centorid_i2e, centorid_es = get_indexed_es(centorid['实体识别结果'])
    data_e2i, data_i2e, data_es = get_indexed_es(data['实体识别结果'])
    
    Errors = get_error_list(centorid_kg,centorid)
    #print(Errors)
    fixs = {}
    data_entity = data['实体识别结果']
    for i in range(len(data_entity)):
        for j in range(len(data_entity)):
            ei = data_i2e[i]
            ej = data_i2e[j]
            ci = -1
            cj = -1
            if ei in centorid_e2i:
                ci = centorid_e2i[ei]
            if ej in centorid_e2i:
                cj = centorid_e2i[ej]
            g = str(ci)+'-'+str(cj) 
            #print(g)
            if g in Errors:
                fixs[str(i)+'-'+str(j)] = Errors[g]
    
    FixedKG = []
    for i,j,v in data_kg:
        g = str(i) + '-' + str(j)
        if g in fixs and fixs[g] == 0:
            continue
        FixedKG.append([i,j,v])
    for f in fixs:
        i,j = f.split('-')
        i = int(i)
        j = int(j)
        FixedKG.append([i,j,4])
    
    return FixedKG


def parse_knowledge_to_numpy(knowledge,max_entity_num):
    z = np.zeros((max_entity_num,max_entity_num))
    for i,j,v in knowledge:
        z[i,j] = v
    return z

def parse_label_to_numpy(labels,max_entity_num):
    z = np.zeros((max_entity_num,max_entity_num))
    for lid in range(len(labels)):
        label = labels[lid]
        for i in range(len(label)-1):
            for e1 in label[i]:
                for e2 in label[i+1]:
                    z[e1,e2] = 1
    return z

#PLAN_TYPE = ['S_DOSAGE','UMETHOD','DURATION','FREQ','DOSAGE','UTIME']

def find_root(tlabel):
    root = []
    for i in range(len(tlabel)):
        if tlabel[:,i].sum()==0 and tlabel[i,:].sum()>0:
            root.append(i)
    return root

def travel(tlabel,index,path,ES,result):
    flag = 0
    #result.append(tail_path)
    for j in range(len(ES)):
        if tlabel[index,j]>0 and not  ES[str(j)][1] in PLAN_TYPE:
            cpath = path.copy()
            cpath.append(j)
            travel(tlabel,j,cpath,ES,result)
    flag = 1
    for j in range(len(ES)):
        if tlabel[index,j]>0 and ES[str(j)][1] in PLAN_TYPE:
            if flag:
                path.append([])
            path[-1].append(j)
            flag = 0
            
    if flag == 0:
        result.append(path)

def after_process(label,ES):
    root = find_root(label)
    result = []
    for v in root:
        travel(label,v,[v],ES,result)
    return result
