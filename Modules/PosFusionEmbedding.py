import torch
from torch import nn
import torch.nn.functional as F

class PosFusionEmbedding(nn.Module):
    def __init__(self, pe, pe_ss, pe_ee, max_seq_len, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.pe_ss = pe_ss
        self.pe_ee = pe_ee
        self.pe = pe

        self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),
                                                nn.ReLU(inplace=True))

    def forward(self, pos_s, pos_e):
        batch = pos_s.size(0)

        pos_ss = pos_s.unsqueeze(-1) - pos_s.unsqueeze(-2)
        pos_ee = pos_e.unsqueeze(-1) - pos_e.unsqueeze(-2)

        max_seq_len = pos_s.size(1)
        pe_ss = self.pe_ss[(pos_ss).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_ee = self.pe_ee[(pos_ee).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])

        pe_2 = torch.cat([pe_ss, pe_ee], dim=-1)
        rel_pos_embedding = self.pos_fusion_forward(pe_2)

        return rel_pos_embedding
