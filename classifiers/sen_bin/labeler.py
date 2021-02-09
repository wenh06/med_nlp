"""
"""
import os
import json
import re
from random import shuffle
from typing import Union, Optional, List, Tuple, Sequence, NoReturn
from numbers import Real, Number
from copy import deepcopy

import numpy as np
from easydict import EasyDict as ED
import ipywidgets as W
import pandas as pd
from IPython.display import display

from etc import ner_const
from utils.cache_utils import load_dosage_output_and_ner
from ner_group.parse.utils_interval import (
    generalized_intervals_intersection,
    generalized_interval_len
)


__all__ = [
    "SentenceBinaryLabeler",
]


class SentenceBinaryLabeler(object):
    """

    check and label validity of images and bounding box annotations for object detection datasets
    """
    __PUNCS__ = ["，", "；", "。", "\[newline\]"]
    __NERNUM__ = [getattr(ner_const, item) for item in dir(ner_const) if item.startswith("NER_NUM_")]
    __MODES__ = ["unique", "full"]
    
    def __init__(self, labeler:str, source_file_prefix:str, save_path:str, l_kgid:Optional[Sequence[str]]=None, batch_size:int=10, neighbor_num:int=3, mode:str="unique", allow_duplicates:bool=True, inspecting:bool=False, **kwargs):
        """
        labeler: 
            打标人代号
        source_file_prefix: 
            数据源文件前缀
        save_path: 
            打标数据保存地址
        l_kgid: 
            需要打标说明书kgid列表
        batch_size: 
            一次打标说明书个数；一个batch完成之后会保存一次
        neighbor_num: 
            展示当前句子的前后文句子数目（左右各`neighbor_num`个）
        mode: 
            模式；"unique"模式下每个说明书只需要一次标注；"full"模式模式下需要每个标注人对每个说明书进行标注
        allow_duplicate: 
            一个说明书是否允许标注多次
        inspecting:
            该状态下，不标注新（未标注）说明书，只检查已标注说明书正确性
        """
        self.labeler = labeler
        self.source_file_prefix = source_file_prefix
        self.save_path = save_path
        self.mode = mode.lower()
        assert self.mode in self.__MODES__
        self.allow_duplicates = allow_duplicates
        self.batch_size = batch_size
        self.shuffle_pending = kwargs.get("shuffle_pending", False)

        self.label_names = {
            "2": "标题",
            "1": "用法用量相关",
            "0": "用法用量无关",
        }
        self.font_color = {
            "2": "green",
            "1": "blue",
            "0": "red",
        }
        self.font_color_chn = {
            "2": "绿色",
            "1": "蓝色",
            "0": "红色",
        }
        self.opposite_label = {
            "1": "0",
            "0": "1",
        }
        self.neighbor_num = neighbor_num

        self.mini_counter = 0
        self.counter = 0
        
        self.current_kgid = None
        self.current_sentences = None
        self.current_punctuations = None
        self.current_raw_label = None
        self.current_label = None

        self.all_kgid = l_kgid
        self.d_text = None
        self.d_ner = None
        self.load_data()

        self.__inspecting = inspecting
        self.__frozen_label_counters = None
        self.__frozen_inspect_counters = None
        self.__frozen_lb_saving = None
        if self.__inspecting:
            raise NotImplementedError("inspect模式还没搞完")

        self.kgid_saved = None
        self.lb_saved = None
        self.pending_kgid = None
        self.update_save_status()

        self.keep_button = W.Button(description="保持标签")
        self.change_button = W.Button(description="改变标签")
        self.Wout = W.Output()

    def load_data(self):
        """
        to write
        """
        self.d_text, self.d_ner = \
            load_dosage_output_and_ner(self.source_file_prefix)
        if self.all_kgid:
            self.d_text = {k: self.d_text[k][1] for k in self.all_kgid}
        else:
            self.d_text = {k: v[1] for k,v in self.d_text.items()}
            self.all_kgid = list(self.d_text.keys())
        save_dir = os.path.dirname(self.save_path)
        os.makedirs(save_dir, exist_ok=True)

    def update_save_status(self):
        """
        to write
        """
        if os.path.isfile(self.save_path):
            with open(self.save_path, "r") as f:
                self.lb_saved = [ED(json.loads(l.strip())) for l in f]
                if self.mode == 'unique':
                    self.kgid_saved = [d.kgid for d in self.lb_saved]
                elif self.mode == 'full':
                    self.kgid_saved = [d.kgid for d in self.lb_saved if d.labeler == self.labeler]
        else:
            self.lb_saved, self.kgid_saved = [], []
        
        self.pending_kgid = list(set(self.all_kgid).difference(set(self.kgid_saved)))
        if self.shuffle_pending:
            shuffle(self.pending_kgid)
        else:
            self.pending_kgid = [item for item in self.all_kgid if item in self.pending_kgid]
        print(f"pending drug instructions updated, current total number is {len(self.pending_kgid)}")

        self.lb_saving = [
            ED(kgid=kgid, file=self.source_file_prefix, label="", labeler=self.labeler) \
                for kgid in self.pending_kgid[:self.batch_size]
        ]
        self.counter = 0
        self.prepare_one_drug_instruction()

    def save_to_file(self):
        """
        to write
        """
        if self.__inspecting:
            raise NotImplementedError
        if os.path.isfile(self.save_path):
            with open(self.save_path, "r") as f:
                self.lb_saved = [ED(json.loads(l.strip())) for l in f]
                if self.allow_duplicates:
                    self.kgid_saved = [d.kgid for d in self.lb_saved if d.labeler==self.labeler]
                else:
                    self.kgid_saved = [d.kgid for d in self.lb_saved]
                self.pending_kgid = list(set(self.all_kgid).difference(set(self.kgid_saved)))
                if self.shuffle_pending:
                    shuffle(self.pending_kgid)
                else:
                    self.pending_kgid = [item for item in self.all_kgid if item in self.pending_kgid]
        else:
            self.lb_saved = []
            self.kgid_saved = []
            self.pending_kgid = deepcopy(self.all_kgid)
            if self.shuffle_pending:
                shuffle(self.pending_kgid)
            else:
                self.pending_kgid = [item for item in self.all_kgid if item in self.pending_kgid]
        print(f"pending drug instructions updated, current total number is {len(self.pending_kgid)}")
        with open(self.save_path, 'a') as f:
            for item in self.lb_saving:
                if item.kgid in self.pending_kgid:
                    f.write(f"{json.dumps(item)}\n")

    def __iter__(self):
        """
        """
        return self

    def __next__(self):
        """
        to write
        """
        if self.__inspecting:
            return self.__next_inspect(simple=True, allow_empty=True)  # 目前只有简单模式
        cond = self.counter < len(self.lb_saving) - 1
        cond = cond or ((self.counter == len(self.lb_saving)-1) and self.mini_counter < len(self.current_sentences))
        if cond:
            print(f"已完成说明书：{self.counter} / {len(self.lb_saving)} ...")

            if len(self.current_sentences) == self.mini_counter:
                self.lb_saving[self.counter].label = "".join(self.current_label)
                self.counter += 1
                self.prepare_one_drug_instruction()
                return self.__next__()
            
            print(f"已完成句子：{self.mini_counter} / {len(self.current_sentences)} ...")
            print(f"目前kgid：{self.current_kgid}")
            text = self._emphasize_brackets(self.d_text[self.current_kgid])
            printmd(f"目前说明书原文：{text}")
            printmd(f"<span style='color:{self.font_color['2']}'>{self.font_color_chn['2']}代表标题</span>, <span style='color:{self.font_color['1']}'>{self.font_color_chn['1']}代表与用法用量相关</span>, <span style='color:{self.font_color['0']}'>{self.font_color_chn['0']}代表与用法用量无关</span>")
            print("\n|\n|\n")

            display_content = ""
            for idx in range(max(0,self.mini_counter-self.neighbor_num), min(self.mini_counter+self.neighbor_num+1,len(self.current_sentences))):
                tmp_str = f"<span style='color:{self.font_color[self.current_label[idx]]}'>{self.current_sentences[idx]}</span>{self.current_punctuations[idx]}"
                if idx == self.mini_counter:
                    tmp_str = f"**<font size='6'>{tmp_str}</font>**"
                display_content += tmp_str
            printmd(display_content)
            self.mini_counter += 1
        else:
            print("one batch finished, validity labels saved and a new batch loaded")
            self.lb_saving[self.counter].label = "".join(self.current_label)
            self.save_to_file()
            self.update_save_status()
            if len(self.pending_kgid) == 0:
                raise StopIteration()
            self.__next__()

    def prepare_one_drug_instruction(self):
        """
        to write
        """
        if self.__inspecting:
            raise NotImplementedError
            return self._prepare_one_drug_instruction_inspect()
        self.prev_label = deepcopy(self.current_label) if self.current_label else None
        self.current_kgid = self.lb_saving[self.counter].kgid
        text = self.d_text[self.current_kgid]
        l_ner = self.d_ner[self.current_kgid]
        ner_intervals = [
            [n.begin, n.end] for n in l_ner if n.get_ner_num() in self.__NERNUM__
        ]
        self.current_sentences = []
        self.current_punctuations = []
        # sentence_intervals = []
        self.current_raw_label = []
        start = 0
        for s in re.finditer("|".join(self.__PUNCS__), text):
            self.current_punctuations.append(s.group())
            end = s.start()
            # sentence_intervals.append([start,end])
            self.current_sentences.append(text[start:end])
            intersection = \
                generalized_intervals_intersection([[start,end]], ner_intervals)
            if generalized_interval_len(intersection) > 0:
                self.current_raw_label.append("1")
            else:
                self.current_raw_label.append("0")
            start = s.end()
        if start < len(text):
            end = len(text)
            self.current_punctuations.append("。")
            self.current_sentences.append(text[start:end])
            intersection = \
                generalized_intervals_intersection([[start,end]], ner_intervals)
            if generalized_interval_len(intersection) > 0:
                self.current_raw_label.append("1")
            else:
                self.current_raw_label.append("0")
        self.current_label = deepcopy(self.current_raw_label)
        self.mini_counter = 0
        self.current_punctuations += ["。"]*(max(0, len(self.current_sentences)-len(self.current_punctuations)))

        if len(self.current_sentences) == 0:
            print("遇到空说明书！")
            self.__next__()

    def change_label(self):
        """
        to write
        """
        if self.__inspecting:
            raise NotImplementedError
        if not 1<=self.mini_counter<=len(self.current_sentences):
            printmd("**<font size='6'><span style='color:red'>不要乱按！</span></font>**")
            return
        if len(self.current_sentences[self.mini_counter-1]) == 0:
            printmd("**<font size='4'><span style='color:red'>连续标点符号或newline带来的空句子，无需改变标签</span></font>**")
            return
        original_label = self.current_raw_label[self.mini_counter-1]
        current_label = self.opposite_label[original_label]
        self.current_label[self.mini_counter-1] = current_label
        msg_color = self.font_color[self.current_label[self.mini_counter-1]]
        msg_content = self.current_sentences[self.mini_counter-1]
        msg = f"**<font size='6'><span style='color:{msg_color}'>{msg_content}</span></font>**从 \042{self.label_names[original_label]}\042 转换为 \042{self.label_names[current_label]}\042"
        printmd(msg)
    
    def set_as_title(self):
        """
        to write
        """
        if self.__inspecting:
            raise NotImplementedError
        if not 1<=self.mini_counter<=len(self.current_sentences):
            printmd("**<font size='6'><span style='color:red'>不要乱按！</span></font>**")
            return
        if len(self.current_sentences[self.mini_counter-1]) == 0:
            printmd("**<font size='4'><span style='color:red'>连续标点符号或newline带来的空句子，无需改变标签</span></font>**")
            return
        self.current_label[self.mini_counter-1] = "2"
        printmd(f"**<font size='6'><span style='color:green'>{self.current_sentences[self.mini_counter-1]}</span></font>**设为标题成功！")

    def rollback(self):
        """
        to write
        """
        if self.__inspecting:
            if self.counter == 0:
                print("已经到头了")
                return
            self.counter -= 1
            self.mini_counter = 0
            print("已回滚至上一说明书")
            return
        if self.mini_counter > 1:
            self.mini_counter -= 2
            self.current_label[self.mini_counter:self.mini_counter+2] = self.current_raw_label[self.mini_counter:self.mini_counter+2]
        elif self.mini_counter == 1:
            self.mini_counter -= 1
            self.current_label[self.mini_counter] = self.current_raw_label[self.mini_counter]
        elif self.counter > 0:
            self.counter -= 1
            print("回滚至上一句")
            prev_label = deepcopy(self.prev_label)
            # ori_lb = deepcopy(self.current_label)
            self.prepare_one_drug_instruction()
            # self.current_raw_label[:-1] = prev_label[:-1]
            self.current_label[:-1] = prev_label[:-1]
            self.mini_counter = len(self.current_sentences)-1
        else:
            printmd("**<font size='6'><span style='color:red'>已写文件，暂时不支持回滚</span></font>**")
            return
        printmd(f"**<font size='6'><span style='color:{self.font_color[self.current_raw_label[self.mini_counter]]}'>成功回滚至第{self.mini_counter}句：{self.current_sentences[self.mini_counter]}</span></font>**")

    def _emphasize_brackets(self, text:str) -> str:
        """
        to write
        """
        new_text = deepcopy(text)
        for s in re.finditer("(\([^\(\)（）]+\))|(（[^\(\)（）]+）)", text):
            new_text = new_text.replace(s.group(), f"<font size='3'><span style='color:red'>{s.group()}</span></font>")
        return new_text

    def view_current(self):
        """
        to write
        """
        print(f"kgid: {self.current_kgid}")
        print(f"text: {self.d_text[self.current_kgid]}")
        print(f"d_ner: {self.d_ner[self.current_kgid]}")
        if self.__inspecting:
            print(f"label: {' | '.join(self.current_raw_label)}")
    
    # def change_button_clicked(self):
    #     """
    #     to write
    #     """
    #     with self.Wout:
    #         self.current_label.append(self.opposite_label[self.current_raw_label[self.mini_counter]])

    def _reset_current(self):
        """
        to write
        """
        self.current_kgid = None
        self.current_raw_label = None
        self.current_label = None
        self.current_punctuations = None
        self.current_sentences = None

    def label(self):
        """
        to write
        """
        if self.__inspecting:
            # currently is inspecting
            self.__frozen_inspect_counters = self.counter, self.mini_counter
            if self.__frozen_label_counters:
                self.counter, self.mini_counter = self.__frozen_label_counters
            else:
                self.counter, self.mini_counter = 0, 0
            self.__frozen_label_counters = None
            self.__frozen_lb_saving = deepcopy(self.lb_saving)
            self._reset_current()
            self.prepare_one_drug_instruction()
        self.__inspecting = False

    def inspect(self):
        """
        to write
        """
        if not self.__inspecting:
            # currently is labeling
            self.__frozen_label_counters = self.counter, self.mini_counter
            if self.__frozen_inspect_counters:
                self.counter, self.mini_counter = self.__frozen_inspect_counters
            else:
                self.counter, self.mini_counter = 0, 0
            self.__frozen_inspect_counters = None
            self.__frozen_lb_saving = deepcopy(self.lb_saving)
            self._reset_current()
            self.prepare_one_drug_instruction()
        self.__inspecting = True

    def __next_inspect(self, simple:bool=True, allow_empty:bool=True):
        """
        simple: 一次以一个DataFrame的形式展示
        """
        if simple:
            if self.counter == len(self.lb_saved):
                raise StopIteration()
            label = self.lb_saved[self.counter].label
            kgid = self.lb_saved[self.counter].kgid
            text = self.d_text[kgid]
            punctuations = []
            sentences = []
            start = 0
            for s_idx, s in enumerate(re.finditer("|".join(self.__PUNCS__), text)):
                end = s.start()
                if not allow_empty and start == end:  # skip empty string
                    start = s.end()
                    continue
                punctuations.append(s.group())
                sentences.append(text[start:end])
                start = s.end()
            if start < len(text):
                end = len(text)
                punctuations.append("。")
                sentences.append(text[start:end])

            filtered_label, filtered_sentences = [], []
            for l, s in zip(label, sentences):
                # if len(s) == 0:
                #     continue
                # if l not in "01":
                #     continue
                filtered_label.append(l)
                filtered_sentences.append(s)
            df = pd.DataFrame()
            df['句子'] = filtered_sentences
            df['标签'] = filtered_label
            # df.style.apply(lambda row: [f'color: {self.font_color[]}'])
            display(df)
            print(kgid)
            print(text)
            print("*"*100)
            self.counter += 1
        else:
            raise NotImplementedError

    def _prepare_one_drug_instruction_inspect(self):
        """
        to write
        """
        if self.counter >= len(self.lb_saved):
            print("**<font size='6'><span style='color:red'>已经加载完所有标注数据</span></font>**")
            return
        self.prev_label = deepcopy(self.current_label) if self.current_label else None
        self.current_kgid = self.lb_saved[self.counter].kgid
        text = self.d_text[self.current_kgid]
        self.current_sentences = []
        self.current_punctuations = []
        # sentence_intervals = []
        start = 0
        for s in re.finditer("|".join(self.__PUNCS__), text):
            self.current_punctuations.append(s.group())
            end = s.start()
            # sentence_intervals.append([start,end])
            self.current_sentences.append(text[start:end])
            start = s.end()
        if start < len(text):
            end = len(text)
            self.current_punctuations.append("。")
            self.current_sentences.append(text[start:end])
        self.current_raw_label = list(self.lb_saved[self.counter].label)
        self.current_label = deepcopy(self.current_raw_label)
        self.mini_counter = 0
        self.current_punctuations += ["。"]*(max(0, len(self.current_sentences)-len(self.current_punctuations)))

        if len(self.current_sentences) == 0:
            print("遇到空说明书！")
            self.__next__()

    def to_training_data(self, features:Optional[List[str]]=None, with_context:bool=True, train_ratio:float=0.8) -> Tuple[Union[List[str], List[List[str]]], ...]:
        """
        to write
        """
        training_data = []
        for idx, kgid in enumerate(self.kgid_saved):
            label = list(self.lb_saved[idx].label)
            text = self.d_text[kgid]
            # sen_len, entity_len, entity_num, es_ratio
            punctuations = []
            sentences = []
            start = 0
            for s_idx, s in enumerate(re.finditer("|".join(self.__PUNCS__), text)):
                end = s.start()
                # if start == end:  # skip empty string
                #     start = s.end()
                #     continue
                punctuations.append(s.group())
                sentences.append(text[start:end])
                start = s.end()
            if start < len(text):
                end = len(text)
                punctuations.append("。")
                sentences.append(text[start:end])
            td = []
            for l, s in zip(label, sentences):
                if len(s) == 0:
                    continue
                if l not in "01":
                    continue
                tmp = [l, s]
                if not with_context:
                    for fn in (features or []):
                        # 暂时没用features
                        fv = 0
                        tmp.append(f"{fv:.8f}")
                td.append("\t".join(tmp))
            if with_context:
                training_data.append(td)
            else:
                training_data += td
        shuffle(training_data)
        n_samples = len(training_data)
        if with_context:
            split_idx = int(train_ratio*n_samples)
            train = training_data[:split_idx]
            test = training_data[split_idx:]
        else:
            positives = [l for l in training_data if l.startswith("1")]
            negatives = [l for l in training_data if l.startswith("0")]
            p_split_idx = int(train_ratio*len(positives))
            n_split_idx = int(train_ratio*len(negatives))
            train = positives[:p_split_idx] + negatives[:n_split_idx]
            test = positives[p_split_idx:] + negatives[n_split_idx:]
            shuffle(train)
            shuffle(test)
        return train, test

    @property
    def class_weight(self) -> np.ndarray:
        """
        """
        class_count = {c:0 for c in "01"}
        for idx, kgid in enumerate(self.kgid_saved):
            label = list(self.lb_saved[idx].label)
            text = self.d_text[kgid]
            # sen_len, entity_len, entity_num, es_ratio
            punctuations = []
            sentences = []
            start = 0
            for s_idx, s in enumerate(re.finditer("|".join(self.__PUNCS__), text)):
                end = s.start()
                # if start == end:  # skip empty string
                #     start = s.end()
                #     continue
                punctuations.append(s.group())
                sentences.append(text[start:end])
                start = s.end()
            if start < len(text):
                end = len(text)
                punctuations.append("。")
                sentences.append(text[start:end])
            td = []
            for l, s in zip(label, sentences):
                if len(s) == 0:
                    continue
                if l not in "01":
                    continue
                class_count[l] += 1
        class_weight = {
            k: sum(class_count.values())/v for k,v in class_count.items()
        }
        class_weight = {
            k: v/(min(class_weight.values())) for k,v in class_weight.items()
        }
        class_weight = np.array([class_weight[k] for k in "01"])
        return class_weight


def printmd(md_str:str) -> NoReturn:
    """ finished, checked,

    printing bold, colored, etc., text

    Parameters:
    -----------
    md_str: str,
        string in the markdown style

    References:
    -----------
    [1] https://stackoverflow.com/questions/23271575/printing-bold-colored-etc-text-in-ipython-qtconsole
    """
    try:
        from IPython.display import Markdown, display
        display(Markdown(md_str))
    except:
        print(md_str)
