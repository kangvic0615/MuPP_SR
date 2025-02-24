# -*- coding: utf-8 -*-
# @Time   : 2020/8/17 19:38
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE:
# @Time   : 2020/8/19, 2020/10/2
# @Author : Yupeng Hou, Yujie Lu
# @Email  : houyupeng@ruc.edu.cn, yujielu1998@gmail.com

r"""
GRU4Rec
################################################

Reference:
    Yong Kiam Tan et al. "Improved Recurrent Neural Networks for Session-based Recommendations." in DLRS 2016.

"""

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class MuPP2(SequentialRecommender):
    r"""MuPP2 is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config, dataset):
        super(MuPP2, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.num_item_representation = config["num_item_representation"]
        self.initializer_range = config["initializer_range"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )

        # if self.embedding_size % self.num_item_representation != 0:
        #     raise ValueError("embedding_size must be divisible by num_item_representation")
        # self.divide_embedding_size = self.embedding_size // self.num_item_representation
        self.representation_layer = nn.ModuleList([nn.Linear(self.embedding_size, self.embedding_size) for _ in range(self.num_item_representation)])

        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        
        batch_size, seq_length = item_seq.size()  # seq_length는 max_seq_length와 동일
              
        # 각 배치에 대한 최대 가능 반복 횟수 계산 (원본 시퀀스를 포함해야 하므로 +1)
        max_repeats = ((seq_length) / (item_seq_len)).floor().clamp(min=1)
        
        # 각 배치에 대한 실제 반복 횟수 생성
        num_repeats = torch.stack([
            torch.randint(1, int(max_rep)+1, (1,), device=item_seq.device)[0]
            for max_rep in max_repeats
        ])
        
        # 블록 인덱스 계산
        position_indices = torch.arange(seq_length, device=item_seq.device).unsqueeze(0)
        block_indices = position_indices // (item_seq_len.unsqueeze(1))
        within_block_pos = position_indices % (item_seq_len.unsqueeze(1))
        
        # 유효한 위치 마스크 생성
        valid_mask = block_indices < num_repeats.unsqueeze(1)
        zero_positions = within_block_pos == item_seq_len.unsqueeze(1)
        
        # 원본 시퀀스에서 가져올 인덱스 계산
        src_indices = within_block_pos % item_seq_len.unsqueeze(1)
        
        # 새로운 시퀀스 생성
        new_seq = torch.where(
            valid_mask & ~zero_positions,
            torch.gather(item_seq, 1, src_indices),
            torch.zeros_like(item_seq)
        )
        # 새로운 시퀀스 길이 계산
        new_seq_len = torch.where(
            num_repeats > 1,
            (((item_seq_len) * num_repeats)).clamp(max=seq_length),
            item_seq_len
        )
        
        # 기본 임베딩 생성
        base_emb = self.item_embedding(new_seq)
        new_emb = torch.zeros_like(base_emb)
        
        # 각 블록에 대한 임베딩 적용
        for i in range(num_repeats.max()):
            block_mask = (block_indices == i) & valid_mask & ~zero_positions
            
            if i == num_repeats.max() - 1:
                # 마지막 블록은 원본 임베딩 사용
                new_emb = torch.where(
                    block_mask.unsqueeze(-1),
                    base_emb,
                    new_emb
                )
            else:
                # representation layer 순환 사용
                layer_idx = i % self.num_item_representation
                transformed_emb = self.representation_layer[layer_idx](base_emb)
                new_emb = torch.where(
                    block_mask.unsqueeze(-1),
                    transformed_emb,
                    new_emb
                )
        
        # padding 위치는 원본 padding 임베딩 사용
        padding_mask = (new_seq == 0).unsqueeze(-1)
        new_emb = torch.where(
            padding_mask,
            self.item_embedding(torch.zeros_like(new_seq)),
            new_emb
        )
        
        item_seq_emb_dropout = self.emb_dropout(new_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, new_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
