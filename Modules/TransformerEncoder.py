from torch import nn

from Modules.PosFusionEmbedding import PosFusionEmbedding
from Modules.TransformerEncoderLayer import TransformerEncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_heads,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 dropout=None, scaled=True, ff_size=-1,
                 max_seq_len=-1, ff_activate='relu',
                 pe=None,
                 pe_ss=None, pe_ee=None,
                 use_pytorch_dropout=True, dataset='weibo'):
        super().__init__()

        self.use_pytorch_dropout = use_pytorch_dropout
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        self.scaled = scaled
        self.ff_activate = ff_activate
        self.dropout = dropout
        self.ff_size = ff_size

        self.pe = pe
        self.pe_ss = pe_ss
        self.pe_ee = pe_ee
        self.four_pos_fusion_embedding = PosFusionEmbedding(self.pe, self.pe_ss,
                                                            self.pe_ee, self.max_seq_len, self.hidden_size)

        self.transformer_layer = TransformerEncoderLayer(hidden_size, num_heads,
                                                         layer_preprocess_sequence,
                                                         layer_postprocess_sequence,
                                                         dropout, scaled, ff_size,
                                                         max_seq_len=max_seq_len,
                                                         ff_activate=ff_activate,
                                                         use_pytorch_dropout=True,
                                                         dataset=dataset,
                                                         )

    def forward(self, query, key, value, seq_len, lex_num=0, pos_s=None, pos_e=None):
        rel_pos_embedding = self.four_pos_fusion_embedding(pos_s, pos_e)

        output = self.transformer_layer(query, key, value, seq_len, lex_num=lex_num,
                                        rel_pos_embedding=rel_pos_embedding)

        return output
