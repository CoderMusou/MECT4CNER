from torch import nn

from Modules.AdaptSelfAttention import AdaptSelfAttention
from Modules.LayerProcess import LayerProcess
from Modules.PositionwiseFeedForward import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 dropout=None, scaled=True, ff_size=-1,
                 max_seq_len=-1,
                 ff_activate='relu',
                 use_pytorch_dropout=True,
                 dataset="weibo"
                 ):
        super().__init__()
        self.use_pytorch_dropout = use_pytorch_dropout
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        self.scaled = scaled
        self.ff_activate = ff_activate
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.ff_size = ff_size

        self.layer_postprocess = LayerProcess(self.layer_postprocess_sequence, self.hidden_size, self.dropout['post'],
                                              self.use_pytorch_dropout)
        self.layer_postprocess1 = LayerProcess(self.layer_postprocess_sequence,self.hidden_size,self.dropout['post'],
                                               self.use_pytorch_dropout)

        self.attn = AdaptSelfAttention(self.hidden_size, self.num_heads,
                                       scaled=self.scaled,
                                       max_seq_len=self.max_seq_len,
                                       attn_dropout=self.dropout['attn'],
                                       use_pytorch_dropout=self.use_pytorch_dropout,
                                       dataset=dataset)

        self.ff = PositionwiseFeedForward([hidden_size, ff_size, hidden_size], self.dropout,
                                          ff_activate=self.ff_activate,
                                          use_pytorch_dropout=self.use_pytorch_dropout)

    def forward(self, query, key, value, seq_len, lex_num=0, rel_pos_embedding=None):

        output = self.attn(query, key, value, seq_len, lex_num=lex_num,
                           rel_pos_embedding=rel_pos_embedding)
        res = self.layer_postprocess(value, output)
        output = self.ff(res)
        output = self.layer_postprocess1(res, output)

        return output
