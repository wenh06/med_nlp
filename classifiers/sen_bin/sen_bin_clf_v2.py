"""
"""
import os
import re
from collections import deque
from itertools import repeat
from copy import deepcopy
from typing import Union, Optional, List, Tuple, Sequence, NoReturn

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import Tensor
from torch import optim
from torch.nn import functional as F
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)
from easydict import EasyDict as ED

from .labeler import SentenceBinaryLabeler


__all__ = [
    "sen_bin_clf",
    "SenBin",
]


__PWD = os.path.dirname(os.path.abspath(__file__))


class SenBin(nn.Module):
    """
    以附带前后各N句话（保持顺序）作为输入的二分类模型
    """
    def __init__(self, vocab_size:int, emb_dim:int, n_classes:int=2, n_nbh:int=2, n_hidden_char:int=150, n_hidden_sen:int=150, config:Optional[dict]=None, emb_matrix:np.ndarray=None):
        """
        emb_matrix of shape (vocab_size, emb_dim)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_char = n_hidden_char
        self.n_hidden_sen = n_hidden_sen
        self.n_classes = n_classes
        self.n_nbh = n_nbh
        self.n_sen = 2*self.n_nbh + 1
        
        default_config = ED(
            char_attention="attn_pool",  # "multihead", None
            char_mask=False,
            sen_attention="context_aware",  # "attn_pool", "context_aware", None
            sen_mask=False,
            vocab=read_bert_word_index(),
        )
        self.config = deepcopy(default_config)
        self.config.update(config or {})

        if emb_matrix is not None:
            assert emb_matrix.shape == (self.vocab_size, self.emb_dim)
            _weight = emb_matrix
        else:
            _weight = None

        self.embed = nn.Embedding(
            self.vocab_size, self.emb_dim,
            _weight=_weight
        )

        self.char_lstm = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.n_hidden_char,
            bidirectional=True,
        )

        # word level attention layer
        # self.att_k = Parameter(torch.Tensor(2*n_hidden_char, 1))
        # self.att_W = Parameter(torch.Tensor(2*n_hidden_char, 2*n_hidden_char))
        # self.att_b = Parameter(torch.Tensor(1, 1))
        if self.config.char_attention is None:
            self.char_atten = None
        elif self.config.char_attention.lower() == "attn_pool":
            self.char_atten = AttentivePooling(
                embed_dim=2*self.n_hidden_char
            )
        elif self.config.char_attention.lower() == "multihead":
            self.char_atten = nn.MultiheadAttention(
                embed_dim=2*self.n_hidden_char, num_heads=1,
            )
        
        self.sen_lstm = nn.LSTM(
            input_size=2*self.n_hidden_char,
            hidden_size=self.n_hidden_sen,
            bidirectional=True,
        )

        # self.sen_atten = nn.MultiheadAttention(2*self.n_hidden_sen, 1)
        if self.config.sen_attention is None:
            self.sen_atten = None
        elif self.config.sen_attention.lower() == "context_aware":
            self.sen_atten = ContextAwareAttention(
                embed_dim=2*self.n_hidden_sen,
            )
        elif self.config.sen_attention.lower() == "attn_pool":
            self.sen_atten = AttentivePooling(
                embed_dim=2*self.n_hidden_sen,
            )
        elif self.config.sen_attention.lower() == "multihead":
            self.sen_atten = nn.MultiheadAttention(
                embed_dim=2*self.n_hidden_sen, num_heads=1,
            )

        self.clf = nn.Linear(
            in_features=4*self.n_hidden_sen,
            out_features=self.n_classes,
        )

    def forward(self, input_text:Tensor, input_sen_len:Optional[Tensor]=None) -> Tensor:
        """
        input_text of shape (batch, n_sen(=self.n_sen), n_char)
        """
        assert input_text.shape[1] == self.n_sen, \
            f"`input_text` should have {self.n_sen} sentences, but got {input_text.shape[1]}"
        # char(word) embedding
        char_emb = self.embed(input_text)  # (batch, n_sen, n_char, n_dim)
        # (batch * n_sen, n_char, n_dim)
        char_emb = char_emb.view(-1, char_emb.shape[-2], char_emb.shape[-1])

        # sentence embedding via bidi-lstm
        # (n_char, batch * n_sen, n_dim)
        char_emb = char_emb.permute(1, 0, 2)
        # (n_char, batch * n_sen, 2*n_hidden)
        sen_emb, _ = self.char_lstm(char_emb)
        # print(f"sen_emb.shape = {sen_emb.shape}")
        if self.config.char_attention:
            if self.config.char_mask:
                raise NotImplementedError
            else:
                mask = None
            # (n_char, batch * n_sen, 2*n_hidden)
            if self.config.char_attention.lower() == "attn_pool":
                # (n_char, batch * n_sen, 2*n_hidden) 
                # -> # (batch * n_sen, n_char, 2*n_hidden)
                sen_emb = sen_emb.permute(1,0,2)
                sen_emb = self.char_atten(
                    sen_emb, attn_mask=mask,
                )  # (batch * n_sen, 2*n_hidden)
                # print(f"sen_emb.shape = {sen_emb.shape}")
            elif self.config.char_attention.lower() == "multihead":
                sen_emb, _ = self.char_atten(
                    sen_emb, sen_emb, sen_emb,
                    attn_mask=mask,
                )
                sen_emb = sen_emb[-1,...]  # (batch * n_sen, 2*n_hidden)
        else:
            sen_emb = sen_emb[-1,...]  # (batch * n_sen, 2*n_hidden)
        
        # print(f"sen_emb.shape = {sen_emb.shape}")
        sen_emb = sen_emb.view(-1, self.n_sen, 2*self.n_hidden_char)
        sen_emb = sen_emb.permute(1, 0, 2)  # (n_sen, batch, 2*n_hidden_char)
        context, _ = self.sen_lstm(sen_emb)  # (n_sen, batch, 2*n_hidden_sen)
        if self.config.sen_attention:
            if self.config.sen_mask:
                raise NotImplementedError
            else:
                mask = None
            if self.config.sen_attention.lower() == "context_aware":
                query = context[self.n_nbh, ...]
                # (n_sen, batch, 2*n_hidden_sen) -> (batch, n_sen, 2*n_hidden_sen)
                context = context.permute(1,0,2)
                context = self.sen_atten(context, query, mask)  # (batch, 2*n_hidden_sen)
            elif self.config.sen_attention.lower() == "attn_pool":
                # (n_sen, batch, 2*n_hidden_sen) -> (batch, n_sen, 2*n_hidden_sen)
                context = context.permute(1,0,2)
                context = self.sen_atten(context, mask)  # (batch, 2*n_hidden_sen)
            elif self.config.sen_attention.lower() == "multihead":
                context, _ = self.sen_atten(
                    context, context, context,
                    attn_mask=mask,
                )  # out shape (n_sen, batch, 2*n_hidden_sen)
                context = context[self.n_nbh, ...]  # (batch, 2*n_hidden_sen)
        else:
            context = context[self.n_nbh, ...]
        # print(f"context.shape = {context.shape}, sen_emb.shape = {sen_emb.shape}")
        fusion = torch.cat([context, sen_emb[self.n_nbh, ...]], dim=-1)
        sen_out = self.clf(fusion)  # (batch, n_classes)

        return sen_out


    def predict_sentences(self, doc:str, gen_config:Optional[ED]=None, bin_clf_thr:float=0.5, with_punc:bool=True, verbose:int=1) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        gen_config contains vocab, max_sen_len, etc
        """
        default_config = ED(
            vocab=self.config.vocab,
            max_sen_len=64,
        )
        default_config.update(gen_config or {})

        sentences = []
        punctuations = []
        start = 0
        for s in re.finditer("|".join(SentenceBinaryLabeler.__PUNCS__), doc):
            punctuations.append(s.group())
            end = s.start()
            # sentence_intervals.append([start,end])
            sentences.append(doc[start:end])
            start = s.end()
        if start < len(doc):
            end = len(doc)
            punctuations.append("。")
            sentences.append(doc[start:end])
        
        punctuations = [
            punctuations[idx] for idx, s in enumerate(sentences) if len(s) > 0
        ]
        sentences = [s for s in sentences if len(s) > 0]
        sentences = list(repeat("", self.n_nbh)) + sentences + list(repeat("", self.n_nbh))

        doc_mat = []
        for n_sen, s in enumerate(sentences[self.n_nbh:-self.n_nbh]):
            start = n_sen
            end = n_sen + 2 * self.n_nbh + 1
            texts = []
            for sen in sentences[start: end]:
                t = sen.lower()
                vec = []
                ct = 0
                for c in t:
                    if c in default_config.vocab:
                        vec.append(default_config.vocab[c])
                        ct += 1
                    if ct == default_config.max_sen_len:
                        break
                # sen_len.append(len(vec))
                vec += list(repeat(0, default_config.max_sen_len-len(vec)))
                texts.append(vec)
            doc_mat.append(texts)
        doc_mat = np.array(doc_mat)

        input_text = torch.from_numpy(doc_mat)
        proba_pred = self.forward(input_text)
        proba_pred = F.softmax(proba_pred, dim=-1)
        proba_pred = proba_pred.cpu().detach().numpy()
        bin_pred = proba_pred.argmax(-1)

        sentences = sentences[self.n_nbh:-self.n_nbh]

        if verbose >= 1:
            import pandas as pd
            from IPython.display import display
            df = pd.DataFrame()
            df['句子'] = sentences
            df['概率(预测1)'] = proba_pred[...,1].flatten()
            display(df)

        if with_punc:
            output_pos = [
                sentences[idx] + punctuations[idx] \
                    for idx in range(len(sentences)) if bin_pred[idx]
            ]
            output_neg = [
                sentences[idx] + punctuations[idx] \
                    for idx in range(len(sentences)) if not bin_pred[idx]
            ]
        else:
            output_pos = [
                sentences[idx] for idx in range(len(sentences)) if bin_pred[idx]
            ]
            output_neg = [
                sentences[idx] for idx in range(len(sentences)) if not bin_pred[idx]
            ]

        return bin_pred, output_pos, output_neg


    def eval_sentences(self, doc_with_label:List[str], gen_config:Optional[ED]=None, bin_clf_thr=0.5, verbose=1) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        gen_config contains vocab, max_sen_len, etc
        """
        default_config = ED(
            vocab=self.config.vocab,
            max_sen_len=64,
        )
        default_config.update(gen_config or {})

        sentences, labels = [], []
        for item in doc_with_label:
            l, s = item.split("\t")
            if len(s) == 0:
                continue
            sentences.append(s)
            labels.append(l)
        sentences = list(repeat("", self.n_nbh)) + sentences + list(repeat("", self.n_nbh))

        doc_mat = []
        for n_sen, s in enumerate(sentences[self.n_nbh:-self.n_nbh]):
            start = n_sen
            end = n_sen + 2 * self.n_nbh + 1
            texts = []
            for sen in sentences[start: end]:
                t = sen.lower()
                vec = []
                ct = 0
                for c in t:
                    if c in default_config.vocab:
                        vec.append(default_config.vocab[c])
                        ct += 1
                    if ct == default_config.max_sen_len:
                        break
                # sen_len.append(len(vec))
                vec += list(repeat(0, default_config.max_sen_len-len(vec)))
                texts.append(vec)
            doc_mat.append(texts)
        doc_mat = np.array(doc_mat)

        input_text = torch.from_numpy(doc_mat)
        proba_pred = self.forward(input_text)
        proba_pred = F.softmax(proba_pred, dim=-1)
        proba_pred = proba_pred.cpu().detach().numpy()
        bin_pred = proba_pred.argmax(-1)

        sentences = sentences[self.n_nbh:-self.n_nbh]

        if verbose >= 1:
            import pandas as pd
            from IPython.display import display
            df = pd.DataFrame()
            df['句子'] = sentences
            df['概率(预测1)'] = proba_pred[...,1].flatten()
            df['预测'] = bin_pred
            df['标签'] = labels
            display(df)

        output_pos = [sentences[idx] for idx in range(len(sentences)) if bin_pred[idx]]
        output_neg = [sentences[idx] for idx in range(len(sentences)) if not bin_pred[idx]]

        return bin_pred, output_pos, output_neg


# class SenBinLoss(nn.Module):
#     """
#     weighted loss
#     """
#     def __init__(self,):
#         """
#         """
#         super().__init__()

#     def forward(self, input:Tensor, target:Tensor) -> Tensor:
#         """
#         """
#         loss = F.cross_entropy(input.view(-1,input.shape[-1]), target.view(-1))
#         return loss


class SenBinGenerator(Dataset):
    """
    """
    def __init__(self, texts:List[List[str]], config:ED) -> NoReturn:
        """
        config contains vocab, n_nbh, max_sen_len, etc
        """
        self.texts = texts
        self.all_sentences = []
        self.doc_numbering = []
        for doc_idx, doc in enumerate(self.texts):
            self.all_sentences += doc
            self.doc_numbering += list(repeat(doc_idx, len(doc)))
        self.doc_numbering = np.array(self.doc_numbering)
        self.config = ED(deepcopy(config))

        self.empty_sentence = list(repeat(0, self.config.max_sen_len))

    def __len__(self) -> int:
        return len(self.all_sentences)

    def __getitem__(self, index:int) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        texts, sen_len = [], []
        current_doc_no = self.doc_numbering[index]
        prev_n = min(self.config.n_nbh, np.sum(self.doc_numbering[:index]==current_doc_no).astype(int))
        next_n = min(self.config.n_nbh, np.sum(self.doc_numbering[index+1:]==current_doc_no).astype(int))

        # current_docs = self.texts[index-prev_n:index+next_n]
        label = int(self.all_sentences[index].split("\t")[0])

        start = index - prev_n
        end = 1 + index + next_n
        texts, sen_len = [], []
        for sen in self.all_sentences[start: end]:
            l, t = sen.split("\t")
            t = t.lower()
            vec = []
            ct = 0
            for c in t:
                if c in self.config.vocab:
                    vec.append(self.config.vocab[c])
                    ct += 1
                if ct == self.config.max_sen_len:
                    break
            sen_len.append(len(vec))
            vec += list(repeat(0, self.config.max_sen_len-len(vec)))
            texts.append(vec)

        original = list(repeat('', self.config.n_nbh-prev_n)) + self.all_sentences[start: end] + list(repeat('', self.config.n_nbh-next_n))
        texts = list(repeat(self.empty_sentence, self.config.n_nbh-prev_n)) + texts + list(repeat(self.empty_sentence, self.config.n_nbh-next_n))
        sen_len = list(repeat(0, self.config.n_nbh-prev_n)) + sen_len + list(repeat(0, self.config.n_nbh-next_n))
        texts = np.array(texts)
        sen_len = np.array(sen_len)
        label = np.array([label])
        ret_val = {
            "original": original,
            "texts": texts,
            "sen_len": sen_len,
            "label": label,
        }
        return ret_val


def train_sen_bin(model:nn.Module, train_texts:List[List[str]], test_texts:List[List[str]], gen_config:dict, device:str='cpu', save_dir:Optional[str]=None) -> nn.Module:
    """
    gen_config contains vocab, n_nbh, max_sen_len, n_epochs, class_weight, etc
    """
    save_prefix = f"SenBin_epoch"

    _REQUIRED_FIELDS = ["vocab", "n_nbh", "max_sen_len", "n_epochs", "class_weight",]
    lack_fieds = set(_REQUIRED_FIELDS) - set(gen_config.keys())
    assert len(lack_fieds) == 0, \
        f"`gen_config` 缺了 {lack_fieds} 这几项"

    _device = torch.device(device)

    train_gen = SenBinGenerator(train_texts, config=gen_config)
    test_gen = SenBinGenerator(test_texts, config=gen_config)

    train_loader = DataLoader(
        dataset=train_gen,
        batch_size=gen_config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        dataset=test_gen,
        batch_size=gen_config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),  # default
        eps=1e-08,  # default
    )
    cw = torch.from_numpy(np.array(gen_config.class_weight, dtype=np.float32))
    criterion = nn.CrossEntropyLoss(weight=cw)

    n_epochs = gen_config.get("n_epochs", 100)
    log_step = 10

    if save_dir:
        saved_models = deque()

    global_step = 0
    log_sep_line = '\n'+'-'*60+'\n'
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        with tqdm(total=len(train_gen), desc=f'Epoch {epoch + 1}/{n_epochs}', ncols=100) as pbar:
            for epoch_step, batch_data in enumerate(train_loader):
                data_texts = batch_data['texts']
                data_sen_len = batch_data['sen_len']
                labels = batch_data['label'].flatten()
                global_step += 1
                data_texts = data_texts.to(device=device)
                data_sen_len = data_sen_len.to(device=device)
                labels = labels.to(device=device)

                optimizer.zero_grad()

                preds = model(data_texts, data_sen_len)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                if global_step % log_step == 0:
                    pbar.set_postfix(**{
                        'loss (batch)': loss.item(),
                    })
                pbar.update(data_texts.shape[0])

            eval_res, eval_inv_res = evaluate_sen_bin(model, val_loader)

            msg = f"""Train step_{global_step}:{log_sep_line}mean epoch_loss :{epoch_loss/len(train_gen)}\nmean test_loss : {eval_res.loss}\n{log_sep_line}\nTest accuracy(1) : {eval_res.accuracy}\nTest precision(1) : {eval_res.precision}\nTest recall(1) : {eval_res.recall}\nTest f1 score(1) : {eval_res.f1}\n{log_sep_line}\nTest accuracy(0) : {eval_inv_res.accuracy}\nTest precision(0) : {eval_inv_res.precision}\nTest recall(0) : {eval_inv_res.recall}\nTest f1 score(0) : {eval_inv_res.f1}"""
            print(msg)

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_suffix = f'epochloss_{epoch_loss:.5f}'
                save_filename = f'{save_prefix}{epoch + 1}_{save_suffix}.pth'
                save_path = os.path.join(save_dir, save_filename)
                torch.save(model.state_dict(), save_path)
                saved_models.append(save_path)
                if len(saved_models) > 20:
                    model_to_remove = saved_models.popleft()
                    try:
                        os.remove(model_to_remove)
                    except:
                        pass
    return model


@torch.no_grad()
def evaluate_sen_bin(model:nn.Module, data_loader:DataLoader) -> Tuple[ED, ED]:
    """
    """
    model.eval()

    all_loss = []
    label_seq = []
    pred_seq = []
    for batch_data in data_loader:
        data_texts = batch_data['texts']
        data_sen_len = batch_data['sen_len']
        labels = batch_data['label'].flatten()
        data_texts = data_texts.to(device=torch.device('cpu'))
        data_sen_len = data_sen_len.to(device=torch.device('cpu'))

        preds = model(data_texts, data_sen_len)
        loss = F.cross_entropy(preds, labels)
        all_loss.append(loss.item())

        bin_preds = F.softmax(preds, dim=-1).cpu().detach().numpy().argmax(-1)

        label_seq += labels.numpy().tolist()
        pred_seq += bin_preds.tolist()

    # print(f"head of label_seq = {label_seq[:5]}, head of pred_seq = {pred_seq[:5]}")

    inv_label_seq, inv_pred_seq = [1-item for item in label_seq], [1-item for item in pred_seq]

    metrics = ED(
        loss=np.mean(all_loss),
        accuracy=accuracy_score(label_seq, pred_seq),
        precision=precision_score(label_seq, pred_seq),
        recall=recall_score(label_seq, pred_seq),
        f1=f1_score(label_seq, pred_seq),
    )

    inv_metrics = ED(
        loss=np.mean(all_loss),
        accuracy=accuracy_score(inv_label_seq, inv_pred_seq),
        precision=precision_score(inv_label_seq, inv_pred_seq),
        recall=recall_score(inv_label_seq, inv_pred_seq),
        f1=f1_score(inv_label_seq, inv_pred_seq),
    )

    return metrics, inv_metrics


class SeqSenBin(nn.Module):
    """
    """
    def __init__(self, vocab_size:int, emb_dim:int, n_classes:int=2, n_hidden_char:int=150, n_hidden_sen:int=150, config:Optional[dict]=None, emb_matrix:np.ndarray=None):
        """
        emb_matrix of shape (vocab_size, emb_dim)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_char = n_hidden_char
        self.n_hidden_sen = n_hidden_sen
        self.n_classes = n_classes
        
        default_config = ED(
            char_attention="attn_pool",  # "multihead", None
            char_mask=False,
            sen_attention="multihead",  # "attn_pool", "context_aware" None
            sen_mask=False,
        )
        self.config = deepcopy(default_config)
        self.config.update(config or {})

        if emb_matrix is not None:
            assert emb_matrix.shape == (self.vocab_size, self.emb_dim)
            _weight = emb_matrix
        else:
            _weight = None

        self.embed = nn.Embedding(
            self.vocab_size, self.emb_dim,
            _weight=_weight
        )

        self.char_lstm = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.n_hidden_char,
            bidirectional=True,
        )

        # word level attention layer
        # self.att_k = Parameter(torch.Tensor(2*n_hidden_char, 1))
        # self.att_W = Parameter(torch.Tensor(2*n_hidden_char, 2*n_hidden_char))
        # self.att_b = Parameter(torch.Tensor(1, 1))
        if self.config.char_attention is None:
            self.char_atten = None
        elif self.config.char_attention.lower() == "attn_pool":
            self.char_atten = AttentivePooling(
                embed_dim=2*self.n_hidden_char
            )
        elif self.config.char_attention.lower() == "multihead":
            self.char_atten = nn.MultiheadAttention(
                embed_dim=2*self.n_hidden_char, num_heads=1,
            )
        
        self.sen_lstm = nn.LSTM(
            input_size=2*self.n_hidden_char,
            hidden_size=self.n_hidden_sen,
            bidirectional=True,
        )

        # self.sen_atten = nn.MultiheadAttention(2*self.n_hidden_sen, 1)
        if self.config.sen_attention is None:
            self.sen_atten = None
        elif self.config.sen_attention.lower() == "context_aware":
            self.sen_atten = ContextAwareAttention(
                embed_dim=2*self.n_hidden_sen,
            )
        elif self.config.sen_attention.lower() == "attn_pool":
            self.sen_atten = AttentivePooling(
                embed_dim=2*self.n_hidden_sen
            )
        elif self.config.sen_attention.lower() == "multihead":
            self.sen_atten = nn.MultiheadAttention(
                embed_dim=2*self.n_hidden_sen, num_heads=1,
            )

        self.clf = nn.Linear(
            in_features=2*self.n_hidden_sen,
            out_features=self.n_classes,
        )

    def forward(self, input_text:Tensor, input_sen_len:Tensor, input_doc_len:Tensor) -> Tensor:
        """
        input_text of shape (n_doc(batch), n_sen, n_char)
        """
        n_doc, n_sen, n_char = input_text.shape

        # char(word) embedding
        char_emb = self.embed(input_text)  # (n_doc(batch), n_sen, n_char, n_dim)
        # (n_doc(batch) * n_sen, n_char, n_dim)
        char_emb = char_emb.view(-1, char_emb.shape[-2], char_emb.shape[-1])

        # sentence embedding via bidi-lstm
        # (n_char, n_doc(batch) * n_sen, n_dim)
        char_emb = char_emb.permute(1, 0, 2)
        # (n_char, n_doc(batch) * n_sen, 2*n_hidden)
        sen_emb, _ = self.char_lstm(char_emb)
        if self.config.char_attention:
            if self.config.char_mask:
                mask = self.sequence_mask(
                    input_sen_len.view(-1),
                    maxlen=n_char,
                    dtype=torch.float32,
                )
            else:
                mask = None
            # (n_char, n_doc(batch) * n_sen, 2*n_hidden)
            if self.config.char_attention.lower() == "attn_pool":
                sen_emb = self.char_atten(
                    sen_emb, attn_mask=mask,
                )
            elif self.config.char_attention.lower() == "multihead":
                sen_emb, _ = self.char_atten(
                    sen_emb, sen_emb, sen_emb,
                    attn_mask=mask,
                )
                sen_emb = sen_emb[-1,...]  # (n_doc(batch) * n_sen, 2*n_hidden)
        else:
            sen_emb = sen_emb[-1,...]  # (n_doc(batch) * n_sen, 2*n_hidden)
        
        # # ? attention ?
        # # h of shape (n_doc(batch) * n_sen, 2*n_hidden)
        # h = F.tanh(torch.matmul(sen_emb, self.att_W) + self.att_b)
        # # h of shape (n_doc(batch) * n_sen, 1)
        # score = torch.matmul(sen_emb, self.att_k)

        sen_emb = sen_emb.view(n_doc, n_sen, 2*self.n_hidden_char)
        sen_emb = sen_emb.permute(1, 0, 2)  # (n_sen, n_doc, 2*n_hidden_char)
        sen_out, _ = self.sen_lstm(sen_emb)  # (n_sen, n_doc, 2*n_hidden_sen)
        if self.config.sen_attention:
            if self.config.sen_mask:
                mask = self.sequence_mask(
                    input_doc_len,
                    maxlen=n_sen,
                    dtype=torch.float32,
                )
                mask = mask.repeat(1,1,n_sen).reshape(n_doc,n_sen,n_sen)
            else:
                mask = None
            if self.config.sen_attention.lower() == "attn_pool":
                raise NotImplementedError
            elif self.config.sen_attention.lower() == "multihead":
                sen_out, _ = self.sen_atten(
                    sen_out, sen_out, sen_out,
                    attn_mask=mask,
                )  # out shape (n_sen, n_doc, 2*n_hidden_sen)
        sen_out = sen_out.permute(1, 0, 2)  # (n_doc, n_sen, 2*n_hidden_sen)
        sen_out = self.clf(sen_out)  # (n_doc, n_sen, n_classes)

        return sen_out

    def sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        """ finished, checked,
        """
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = (row_vector < matrix)
        mask = mask.to(dtype)
        return mask

    def predict_sentences(self, doc:List[str], gen_config:ED, bin_clf_thr=0.5, verbose=1):
        """
        """
        sentences = []
        punctuations = []
        start = 0
        for s in re.finditer("|".join(SentenceBinaryLabeler.__PUNCS__), doc):
            punctuations.append(s.group())
            end = s.start()
            # sentence_intervals.append([start,end])
            sentences.append(doc[start:end])
            start = s.end()
        if start < len(doc):
            end = len(doc)
            punctuations.append("。")
            sentences.append(doc[start:end])
        
        sentences = [s for s in sentences if len(s) > 0]

        doc_mat, sen_len = [], []
        for n_sen, s in enumerate(sentences):
            s = s.lower()
            vec = []
            cs = 0
            for c in s:
                if c in gen_config.vocab:
                    vec.append(gen_config.vocab[c])
                    cs += 1
                if cs == gen_config.max_sen_len:
                    break
            sen_len.append(len(vec))
            vec += list(repeat(0, gen_config.max_sen_len-len(vec)))
            doc_mat.append(vec)

            if n_sen + 1 == gen_config.max_sen_num:
                    break
            
        doc_mat += list(repeat(list(repeat(0, gen_config.max_sen_len)), gen_config.max_sen_num-len(doc_mat)))
        sen_len += list(repeat(0, gen_config.max_sen_num-len(sen_len)))

        doc_mat = np.array([doc_mat])
        sen_len = np.array([sen_len])
        sen_num = np.array([len(sentences)])

        input_text = torch.from_numpy(doc_mat)
        input_sen_len = torch.from_numpy(sen_len)
        input_doc_len = torch.from_numpy(sen_num)

        proba_pred = self.forward(input_text, input_sen_len, input_doc_len)
        proba_pred = F.softmax(proba_pred, dim=-1)
        proba_pred = proba_pred.cpu().detach().numpy()[0,:len(sentences)]
        bin_pred = proba_pred.argmax(-1)

        if verbose >= 1:
            import pandas as pd
            from IPython.display import display
            df = pd.DataFrame()
            df['句子'] = sentences
            df['概率'] = proba_pred[...,1].flatten()
            df['预测'] = bin_pred
            display(df)

        output_pos = [sentences[idx] for idx in range(len(sentences)) if bin_pred[idx]]
        output_neg = [sentences[idx] for idx in range(len(sentences)) if not bin_pred[idx]]

        return output_pos, output_neg


class AttentivePooling(nn.Module):
    """
    """
    def __init__(self, embed_dim:int, mid_dim:Optional[int]=None, dropout:float=0.2) -> NoReturn:
        """
        """
        super().__init__()
        self.__embed_dim = embed_dim
        self.__mid_dim = mid_dim or self.__embed_dim//2
        self.__dropout = dropout

        self.dropout = nn.Dropout(self.__dropout, inplace=False)
        self.mid_linear = nn.Linear(self.__embed_dim, self.__mid_dim, bias=True)
        self.tanh = nn.Tanh()
        self.contraction = nn.Linear(self.__mid_dim, 1, bias=False)
        self.softmax = nn.Softmax(-1)

    def forward(self, input:Tensor, attn_mask:Optional[Tensor]=None) -> Tensor:
        """
        input of shape (batch_size, seq_len, n_channels)
        """
        scores = self.dropout(input)
        scores = self.mid_linear(scores)  # -> (batch_size, seq_len, n_channels)
        scores = self.tanh(scores)  # -> (batch_size, seq_len, n_channels)
        scores = self.contraction(scores)  # -> (batch_size, seq_len, 1)
        scores = scores.squeeze(-1)  # -> (batch_size, seq_len)
        scores = self.softmax(scores)  # -> (batch_size, seq_len)
        weighted_input = \
            input * (scores[..., np.newaxis]) # -> (batch_size, seq_len, n_channels)
        output = weighted_input.sum(1)  # -> (batch_size, n_channels)
        return output


class ContextAwareAttention(nn.Module):
    """
    """
    def __init__(self, embed_dim:int, dropout:float=0.2):
        """
        """
        super().__init__()
        self.__embed_dim = embed_dim
        self.__dropout = dropout

        self.dropout = nn.Dropout(self.__dropout, inplace=False)
        self.linear = nn.Linear(self.__embed_dim, self.__embed_dim, bias=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(-1)

    def forward(self, input_texts:Tensor, query:Tensor, attn_mask:Optional[Tensor]=None) -> Tensor:
        """
        input_texts of shape (batch_size, seq_len, n_channels)
        query of shape (batch_size, n_channels)
        """
        scores = self.dropout(input_texts)
        scores = self.linear(scores)  # -> (batch_size, seq_len, n_channels)
        scores = self.tanh(scores)  # -> (batch_size, seq_len, n_channels)
        scores = scores * query.unsqueeze(1)  # -> (batch_size, seq_len, n_channels)
        scores = scores.sum(-1)  # -> (batch_size, seq_len, 1)
        scores = self.softmax(scores)  # -> (batch_size, seq_len)
        weighted_input = \
            input_texts * (scores.unsqueeze(-1)) # -> (batch_size, seq_len, n_channels)
        output = weighted_input.sum(1)  # -> (batch_size, n_channels)
        return output


@DeprecationWarning
class Attention(nn.Module):
    """
    """
    def __init__(self, embed_dim:int, bias:bool=True, retseq:bool=False):
        """
        """
        super().__init__()
        self.__embed_dim = embed_dim
        self.__retseq = retseq
        self.linear = nn.Sequential(
            nn.Linear(self.__embed_dim, self.__embed_dim, bias=bias),
            nn.Tanh(),
        )
        if not self.__retseq:
            self.att_k = Parameter(torch.Tensor(self.__embed_dim))
            torch.nn.init.constant_(self.att_k, 1)
        else:
            raise NotImplementedError
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, input:Tensor, mask:Optional[Tensor]=None) -> Tensor:
        """
        input shape: (seq_len, batch_size, embed_dim)
        mask shape: (batch_size, seq_len)
        output shape: (batch_size, embed_dim)
        """
        # (seq_len, batch, embed_dim) -> (batch, seq_len, embed_dim)
        _input = input.permute(1,0,2)
        lin_out = self.linear(_input)  # out shape (batch, seq_len, embed_dim)
        if self.__retseq:
            score = torch.tensordot(lin_out, _input, dims=[[2],[2]])
        else:
            score = torch.tensordot(lin_out, self.att_k, dims=1)
        if mask is None:
            score_softmax = self.softmax(score)
        else:
            score_exp = torch.exp(score)
            score_masked = score_exp * mask
            score_softmax = score_masked / (torch.sum(score_masked, dim=-1, keepdim=True) + 1e-4)
                # + torch.finfo(torch.float32).eps
        output = _input * score_softmax[..., np.newaxis]
        output = output.sum(dim=1)
        return output


class SeqSenBinLoss(nn.Module):
    """
    """
    def __init__(self,):
        """
        """
        super().__init__()

    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        """
        """
        loss = F.cross_entropy(input.view(-1,input.shape[-1]), target.view(-1))
        return loss


class SeqSenBinGenerator(Dataset):
    """
    """
    def __init__(self, texts:List[List[str]], config:ED) -> NoReturn:
        """
        config contains vocab, batch_size, max_sen_len, max_sen_num, etc
        """
        self.texts = texts
        self.config = ED(deepcopy(config))
        nb, res = divmod(len(self.texts), self.config.batch_size)
        self._n_batch = nb if res == 0 else nb+1

    def __len__(self) -> int:
        return self._n_batch

    def __getitem__(self, index:int) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        texts, sen_len, doc_len, labels = [], [], [], []
        start = index * self.config.batch_size
        end = (index+1) * self.config.batch_size
        for doc in self.texts[start: end]:
            doc_len.append(len(doc))
            doc_mat, doc_sen_len, doc_lb = [], [], []
            for n_sen, sen in enumerate(doc):
                l, t = sen.split("\t")
                t = t.lower()
                vec = []
                ct = 0
                for c in t:
                    if c in self.config.vocab:
                        vec.append(self.config.vocab[c])
                        ct += 1
                    if ct == self.config.max_sen_len:
                        break
                doc_sen_len.append(len(vec))
                vec += list(repeat(0, self.config.max_sen_len-len(vec)))
                doc_mat.append(vec)
                doc_lb.append(int(l))
                if n_sen + 1 == self.config.max_sen_num:
                    break
            
            doc_mat += list(repeat(list(repeat(0, self.config.max_sen_len)), self.config.max_sen_num-len(doc_mat)))
            doc_lb += list(repeat(0, self.config.max_sen_num-len(doc_lb)))
            doc_sen_len += list(repeat(0, self.config.max_sen_num-len(doc_sen_len)))
            texts.append(doc_mat)
            sen_len.append(doc_sen_len)
            labels.append(doc_lb)
        texts = np.array(texts)
        doc_len = np.array(doc_len)
        sen_len = np.array(sen_len)
        labels = np.array(labels)
        ret_val = {
            "original": self.texts[start: end],
            "texts": texts,
            "sen_len": sen_len,
            "doc_len": doc_len,
            "labels": labels,
        }
        return ret_val


def train_seq_sen_bin(model:nn.Module, train_texts:List[List[str]], test_texts:List[List[str]], gen_config:dict, device:str='cpu', save_dir:Optional[str]=None) -> nn.Module:
    """
    gen_config contains vocab, batch_size, max_sen_len, max_sen_num
    """
    save_prefix = f"SeqSenBin_epoch"

    _device = torch.device(device)

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),  # default
        eps=1e-08,  # default
    )
    criterion = SeqSenBinLoss()

    n_epochs = 100
    log_step = 10

    if save_dir:
        saved_models = deque()

    global_step = 0
    log_sep_line = '\n'+'-'*60+'\n'
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0
            train_gen = SeqSenBinGenerator(train_texts, config=gen_config)
            test_gen = SeqSenBinGenerator(test_texts, config=gen_config)

            with tqdm(total=len(train_gen), desc=f'Epoch {epoch + 1}/{n_epochs}', ncols=100) as pbar:
                for epoch_step, idx in enumerate(range(len(train_gen))):
                    data = train_gen[idx]
                    data_texts = data['texts']
                    data_sen_len = data['sen_len']
                    data_doc_len = data['doc_len']
                    labels = data['labels']
                    global_step += 1
                    data_texts = torch.from_numpy(data_texts).to(device=device)
                    data_sen_len = torch.from_numpy(data_sen_len).to(device=device)
                    data_doc_len = torch.from_numpy(data_doc_len).to(device=device)
                    labels = torch.from_numpy(labels).to(device=device)

                    optimizer.zero_grad()

                    preds = model(data_texts, data_sen_len, data_doc_len)
                    loss = criterion(preds, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                    if global_step % log_step == 0:
                        pbar.set_postfix(**{
                            'loss (batch)': loss.item(),
                        })
                    pbar.update(1)

                eval_res, eval_inv_res = evaluate_seq_sen_bin(model, test_gen)

                msg = f"""Train step_{global_step}:{log_sep_line}mean epoch_loss :{epoch_loss/len(train_gen)}\nmean test_loss : {eval_res.loss}\nTest accuracy : {eval_res.accuracy}\nTest precision : {eval_res.precision}\nTest recall : {eval_res.recall}\nTest f1 score : {eval_res.f1}"""
                print(msg)

                if save_dir:
                    save_suffix = f'epochloss_{epoch_loss:.5f}'
                    save_filename = f'{save_prefix}{epoch + 1}_{save_suffix}.pth'
                    save_path = os.path.join(save_dir, save_filename)
                    torch.save(model.state_dict(), save_path)
                    saved_models.append(save_path)
                    if len(saved_models) > 20:
                        model_to_remove = saved_models.popleft()
                        try:
                            os.remove(model_to_remove)
                        except:
                            pass
    return model


@torch.no_grad()
def evaluate_seq_sen_bin(model:nn.Module, data_generator:SenBinGenerator) -> Tuple[ED, ED]:
    """
    """
    model.eval()

    all_loss = []
    label_seq = []
    pred_seq = []
    for idx in range(len(data_generator)):
        data = data_generator[idx]
        data_texts = data['texts']
        data_sen_len = data['sen_len']
        data_doc_len = data['doc_len']
        labels = data['labels']
        data_texts = torch.from_numpy(data_texts).to(device=torch.device('cpu'))
        data_sen_len = torch.from_numpy(data_sen_len).to(device=torch.device('cpu'))
        data_doc_len = torch.from_numpy(data_doc_len).to(device=torch.device('cpu'))

        preds = model(data_texts, data_sen_len, data_doc_len)
        loss = F.cross_entropy(preds.view(-1,preds.shape[-1]), torch.from_numpy(labels).view(-1))
        all_loss.append(loss.item())

        bin_preds = F.softmax(preds, dim=-1).cpu().detach().numpy().argmax(-1)

        
        for idx, sen_num in enumerate(data_doc_len):
            label_seq += labels[idx, :sen_num].tolist()
            pred_seq += bin_preds[idx, :sen_num].tolist()

    # print(f"head of label_seq = {label_seq[:5]}, head of pred_seq = {pred_seq[:5]}")

    inv_label_seq, inv_pred_seq = [1-item for item in label_seq], [1-item for item in pred_seq]

    metrics = ED(
        loss=np.mean(all_loss),
        accuracy=accuracy_score(label_seq, pred_seq),
        precision=precision_score(label_seq, pred_seq),
        recall=recall_score(label_seq, pred_seq),
        f1=f1_score(label_seq, pred_seq),
    )

    inv_metrics = ED(
        loss=np.mean(all_loss),
        accuracy=accuracy_score(inv_label_seq, inv_pred_seq),
        precision=precision_score(inv_label_seq, inv_pred_seq),
        recall=recall_score(inv_label_seq, inv_pred_seq),
        f1=f1_score(inv_label_seq, inv_pred_seq),
    )

    return metrics, inv_metrics


def read_bert_word_index(bert_word_index_path:Optional[str]=None):
    """
    """
    default_path = os.path.join(__PWD, "vocab.txt")
    with open(bert_word_index_path or default_path, "r") as f:
        lines=f.readlines()
    index=0
    bert_word_index={}
    for l in lines:
        l=l.strip()
        bert_word_index[l]=index
        index+=1
    return bert_word_index



model_path = os.path.join(__PWD, "pytorch_sen_bin_clf.pth")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model = SenBin(21128, 512)
model.load_state_dict(torch.load(model_path, map_location=device))


def sen_bin_clf(doc:str, bin_clf_thr:float=0.5, with_punc:bool=True) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    """
    
    bin_pred, positives, negatives = \
        model.predict_sentences(
            doc=doc,
            bin_clf_thr=bin_clf_thr,
            with_punc=with_punc,
            verbose=0,
        )
    return bin_pred, positives, negatives
