import keras
from keras import Input
from keras.layers import *
import keras.backend as K
#from gensim.models import Word2Vec
from keras.optimizers import  *
# import keras_bert

def get_classifier(dim):
    vec_input = Input(shape=(dim,))
    vec = Dense(256,activation='relu')(vec_input)
    #vec = Dense(256,activation='relu')(vec)
    #vec = Dense(128,activation='relu')(vec)
    logit = Dense(1,activation='sigmoid')(vec)
    model = keras.Model(vec_input,logit)
    return model

def MyLoss(ytrue,ypred):
    logit, mask = ypred

    logit = ytrue*K.log(logit)+(1-ytrue)*K.log(1-logit)
    logit = logit*mask
    logit = K.sum(logit,)
    
    return logit

from keras.layers.core import Lambda

class RemoveMask(Lambda):
    def __init__(self):
        super(RemoveMask, self).__init__((lambda x, mask: x))
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return None

def get_model(max_text_length,max_entity_num,char_num,tp_dict):
    texts_input = Input(shape=(max_text_length,),dtype='int32')
    entity_input = Input(shape=(max_entity_num,max_text_length),dtype='float32')
    entity_type_input = Input(shape=(max_entity_num,),dtype='int32')
    knowledge_input = Input(shape=(max_entity_num,max_entity_num),dtype='int32')
    #mask_matrix_input = Input(shape=(max_entity_length,max_entity_length))

    embedding_layer = Embedding(char_num+1,200,trainable=True)
    text_embedding = embedding_layer(texts_input)

    text_embedding = Dropout(0.2)(text_embedding)
    text_embedding = Dense(400)(text_embedding)

    text_vecs = Conv1D(400,kernel_size=3,activation='relu',padding='same')(text_embedding)
    text_vecs = Bidirectional(LSTM(200,return_sequences=True))(text_vecs) #(max_text_length,400)
    
    entity_type_embedding_layer = Embedding(len(tp_dict),200,trainable=True)  # NOTE: bug here
    entity_type_emb = entity_type_embedding_layer(entity_type_input)
    entity_type_emb = Dropout(0.2)(entity_type_emb)

    entity_emb = keras.layers.Dot(axes=[-1,-2])([entity_input,text_vecs]) #(max_entity_length,400)
    entity_emb = keras.layers.Concatenate(axis=-1)([entity_emb,entity_type_emb])
    entity_emb = Dense(400)(entity_emb)
    entity_vecs = Conv1D(400,kernel_size=3,activation='relu',padding='same')(entity_emb)
    entity_vecs = Bidirectional(LSTM(200,return_sequences=True))(entity_vecs)
    entity_vecs = keras.layers.Reshape((max_entity_num,400))(entity_vecs)

    entity_vecs1 = keras.layers.TimeDistributed(RepeatVector(max_entity_num))(entity_vecs)
    entity_vecs2 = Reshape((max_entity_num*400,))(entity_vecs)
    entity_vecs2 = RepeatVector(max_entity_num)(entity_vecs2)
    entity_vecs2 = Reshape((max_entity_num,max_entity_num,400,))(entity_vecs2)

    entity_vecs0 = Concatenate(axis=-1)([entity_vecs1,entity_vecs2])

    knowledge_embedding_layer = Embedding(10,200,trainable=True)
    knowledge_vecs = knowledge_embedding_layer(knowledge_input)
    knowledge_vecs = Dropout(0.3)(knowledge_vecs)
    entity_vecs0 = Concatenate(axis=-1)([entity_vecs0,knowledge_vecs])

    classifier = get_classifier(400*2+200)
    pred = keras.layers.TimeDistributed(keras.layers.TimeDistributed(classifier))(entity_vecs0)
    pred = Reshape((max_entity_num*max_entity_num,1))(pred)

    return keras.Model([texts_input,entity_input,entity_type_input,knowledge_input],pred)
