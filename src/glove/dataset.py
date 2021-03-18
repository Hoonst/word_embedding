from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader

class GloveDataset(Dataset):
    def __init__(self, data_dir: str, n_words=20000, window_size=5):
        with open(data_dir) as f:
            text = f.read()
        self._window_size = window_size
        self._tokens = text.split(" ")[:n_words]
        word_counter = Counter()
        word_counter.update(self._tokens)
        self._word2id = {w:i for i, (w,_) in enumerate(word_counter.most_common())}
        self._id2word = {i:w for w, i in self._word2id.items()}
        self._vocab_len = len(self._word2id)

        self._id_tokens = [self._word2id[w] for w in self._tokens]

        self._create_coocurrence_matrix()

        print("# of words: {}".format(len(self._tokens)))
        print("Vocabulary length: {}".format(self._vocab_len))

    def _create_coocurrence_matrix(self):
        cooc_mat = defaultdict(Counter)
        for i, w in enumerate(self._id_tokens):
            start_i = max(i - self._window_size, 0)
            end_i = min(i + self._window_size + 1, len(self._id_tokens))
            for j in range(start_i, end_i):
                if i != j:
                    c = self._id_tokens[j]
                    # abs로 나누는 이유는 아마 거리에 따른 영향력을 반영하려는 것이 아닐까 싶다.
                    cooc_mat[w][c] += 1 / abs(j-i)

        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()

        #Create indexes and x values tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(v)

        self._i_idx = torch.LongTensor(self._i_idx)
        self._j_idx = torch.LongTensor(self._j_idx)
        self._xij = torch.FloatTensor(self._xij)
        # _xij는 빈도
        # _i_idx / _j_idx 는 index

    def __len__(self):
        return len(self._i_idx)
    
    def __getitem__(self, idx):
        return self._xij[idx], self._i_idx[idx], self._j_idx[idx]

def get_trn_loader(
    data_dir: str,
    batch_size: int,
    n_words: int, 
    window_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = False,
):
    train_dataset = GloveDataset(
        data_dir, n_words = n_words, window_size = window_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        
    )

    return train_loader, train_dataset._vocab_len


