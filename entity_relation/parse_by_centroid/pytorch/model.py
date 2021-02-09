"""
"""
from typing import NoReturn

import numpy as np
import torch
from torch import nn
from torch import functional as F


class Model(nn.Module):
    """
    """
    def __init__(self, max_text_length:int, max_entity_num:int, char_num:int, tp_dict:dict) -> NoReturn:
        """
        """
        self.max_text_length = max_text_length
        self.max_entity_num = max_entity_num
        self.char_num = char_num
        self.tp_dict = tp_dict

        self.text_embedding = nn.Sequential()
        self.text_embedding.add_module(
            "init_embedding",
            nn.Embedding(self.char_num, 200),
        )
        self.text_embedding.add_module(
            "linear",
            nn.Linear(200, 400),
        )
        self.text_embedding.add_module(
            "conv",
            nn.Conv1d(400, 400, 3, padding=1),
        )
        self.text_embedding.add_module(
            "relu",
            nn.ReLU(inplace=True),
        )
        
        self.text_lstm = nn.LSTM(
            400, 200, bidirectional=True,
        )

        self.et_embedding = nn.Embedding(len(tp_dict)+1, 200)

        self.main_stream = nn.Sequential()
        self.ms_lstm = nn.LSTM()
        