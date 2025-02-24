"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class MuPP(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(MuPP, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.num_item_representation = config["num_item_representation"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        
        # if self.hidden_size % self.num_item_representation != 0:
        #     raise ValueError("hidden_size must be divisible by num_item_representation")
        # self.divide_hidden_size = self.hidden_size // self.num_item_representation
        # self.representation_layer = nn.ModuleList([nn.Linear(self.divide_hidden_size, self.hidden_size) for _ in range(self.num_item_representation)])
        self.representation_layer = nn.ModuleList([
        nn.Linear(self.hidden_size, self.hidden_size) 
            for _ in range(self.num_item_representation)
        ])
        
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

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
        
        input_emb = new_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

     
        extended_attention_mask = self.get_attention_mask(new_seq)
        
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, new_seq_len - 1)
        return output

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
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
