from torch import nn

from Modules.MyDropout import MyDropout


class LayerProcess(nn.Module):
    def __init__(self, process_sequence, hidden_size, dropout=0,
                 use_pytorch_dropout=True):
        super().__init__()
        self.use_pytorch_dropout = use_pytorch_dropout
        self.process_sequence = process_sequence.lower()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        if 'd' in self.process_sequence:
            if self.use_pytorch_dropout:
                self.dropout = nn.Dropout(dropout)
            else:
                self.dropout = MyDropout(dropout)
        if 'n' in self.process_sequence:
            self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, res, inp):
        output = inp
        for op in self.process_sequence:
            if op == 'a':
                output = res + inp
            if op == 'd':
                output = self.dropout(output)
            if op == 'n':
                output = self.layer_norm(output)

        return output
