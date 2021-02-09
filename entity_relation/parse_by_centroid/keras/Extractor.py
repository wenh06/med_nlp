from Preprocess import *
from Model import *

class EXTRACTOR:
    
    def __init__(self,meta_data_path):
        self.meta_data_path = meta_data_path
        
        with open(os.path.join(meta_data_path,'char_dict.json')) as f:
            self.char_dict = json.load(f)
        with open(os.path.join(meta_data_path,'tp_dict.json')) as f:
            self.tp_dict = json.load(f)
        with open(os.path.join(meta_data_path,'bert_voca.json')) as f:
            self.bert_voca = json.load(f)
        self.max_text_length = 500
        self.max_entity_num = 100
        
        self.load_model()
        
    def load_model(self,):
        self.model = get_model(self.max_text_length,self.max_entity_num,len(self.char_dict),self.tp_dict)
        self.model.load_weights(os.path.join(self.meta_data_path,'model_weights.h5'))
            
    def preprocess(self,data,centroid):

        knowledge = GetKnowledge(data,centroid)
        knowledge = parse_knowledge_to_numpy(knowledge,self.max_entity_num)
        
        pmask = np.zeros((self.max_entity_num,self.max_entity_num))
        num = len(data['实体识别结果'])
        pmask[:num,:num] = 1

        # 文本
        text = []
        bert_text = []
        for char in data['原文']:
            if char in self.char_dict:
                text.append(self.char_dict[char])
            else:
                text.append(0)
            if char in self.bert_voca:
                bert_text.append(self.bert_voca[char])
            else:
                bert_text.append(0)

        if len(text)<self.max_text_length:
            text += [0]*(self.max_text_length-len(text))
            bert_text += [0]*((self.max_text_length-len(bert_text)))
        else:
            text = text[:self.max_text_length]
            bert_text = bert_text[:self.max_text_length]
            
            
        #parse entity
        e_texts = []
        e_tps = []
        for e in range(len(data['实体识别结果'])):
            e = str(e)
            word,tp,sp,ep = data['实体识别结果'][e]
            g = np.zeros((self.max_text_length,))
            g[sp:ep] = 1/(ep-sp)
            e_texts.append(g)
            if not tp in self.tp_dict:
                e_tps.append(0)
            else:
                e_tps.append(self.tp_dict[tp])

        e_texts = e_texts + [np.zeros((self.max_text_length,))]*(self.max_entity_num-len(e_texts))
        e_tps = e_tps + [0]*(self.max_entity_num-len(e_tps))
        
        text = np.array([text])
        bert_text = np.array([bert_text])
        e_texts = np.array([e_texts])
        e_tps = np.array([e_tps])
        knowledge = np.array([knowledge])
        
        return text,e_texts,e_tps,knowledge>0,pmask
    
    def __call__(self,data,centorid):
        text,e_texts,e_tps,knowledge,pmask = self.preprocess(data,centorid)
        predict_label = self.model.predict([text,e_texts,e_tps,knowledge])
        #predict_label = knowledge
        predict_label = predict_label.reshape((self.max_entity_num,self.max_entity_num))>0.5
        result = after_process(predict_label,data['实体识别结果'])
        return predict_label,result,data['实体识别结果']
