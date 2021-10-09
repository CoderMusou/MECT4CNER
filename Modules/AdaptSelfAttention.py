import math

import torch
import torch.nn.functional as F
from fastNLP import seq_len_to_mask
from torch import nn

from Modules.MyDropout import MyDropout


class AdaptSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads,
                 scaled=True, max_seq_len=-1,
                 attn_dropout=None,
                 use_pytorch_dropout=True, dataset='weibo'):
        super().__init__()
        self.use_pytorch_dropout = use_pytorch_dropout
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        self.max_seq_len = max_seq_len

        assert (self.per_head_size * self.num_heads == self.hidden_size)

        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
        self.u = nn.Parameter(torch.Tensor(self.num_heads, self.per_head_size))
        self.v = nn.Parameter(torch.Tensor(self.num_heads, self.per_head_size))

        if self.use_pytorch_dropout:
            self.dropout = nn.Dropout(attn_dropout)
        else:
            self.dropout = MyDropout(attn_dropout)


        if dataset == 'weibo':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 320, 320), requires_grad=True)
        if dataset == 'msra':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 310, 310), requires_grad=True)
        if dataset == 'resume':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 344, 344), requires_grad=True)
        if dataset == 'ontonotes':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 477, 477), requires_grad=True)
        if dataset == 'tc':
            self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, 398, 398), requires_grad=True)

        nn.init.kaiming_normal_(self.randomAttention, a=math.sqrt(5))

    def forward(self, query, key, value, seq_len, lex_num, rel_pos_embedding):

        query = self.w_q(query)
        value = self.w_v(value)
        rel_pos_embedding = self.w_r(rel_pos_embedding)

        batch = key.size(0)
        max_seq_len = key.size(1)

        # batch * seq_len * n_head * d_head
        key = torch.reshape(key, [batch, max_seq_len, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, max_seq_len, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, max_seq_len, self.num_heads, self.per_head_size])
        rel_pos_embedding = torch.reshape(rel_pos_embedding,
                                          [batch, max_seq_len, max_seq_len, self.num_heads, self.per_head_size])

        # batch * n_head * seq_len * d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        # batch * n_head * d_head * key_len
        key = key.transpose(-1, -2)
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)

        query_and_u_for_c = query + u_for_c

        A_C = torch.matmul(query_and_u_for_c, key)

        #B
        rel_pos_embedding_for_b = rel_pos_embedding.permute(0, 3, 1, 4, 2)
        query_for_b = query.view([batch, self.num_heads, max_seq_len, 1, self.per_head_size])

        query_for_b_and_v_for_d = query_for_b + self.v.view(1, self.num_heads, 1, 1, self.per_head_size)
        B_D = torch.matmul(query_for_b_and_v_for_d, rel_pos_embedding_for_b).squeeze(-2)

        attn_score_raw = A_C + B_D + self.randomAttention[:, :, :max_seq_len, :max_seq_len]

        mask = seq_len_to_mask(seq_len+lex_num).bool().unsqueeze(1).unsqueeze(1)
        attn_score_raw_masked = attn_score_raw.masked_fill(~mask, -1e15)

        attn_score = F.softmax(attn_score_raw_masked, dim=-1)
        attn_score = self.dropout(attn_score)

        value_weighted_sum = torch.matmul(attn_score, value)

        result = value_weighted_sum.transpose(1, 2).contiguous(). \
            reshape(batch, max_seq_len, self.hidden_size)

        return result