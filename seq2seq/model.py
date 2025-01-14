# import torch
# import torch.nn as nn
# import random
# class Encoder(nn.Module):
#     def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
#         super().__init__()
#         self.embedding = nn.Embedding(input_dim + 2, emb_dim)  # Add 2 for <sos> and <eos>
#         self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, src):
#         embedded = self.dropout(self.embedding(src))
#         outputs, (hidden, cell) = self.rnn(embedded)
#         return hidden, cell

# class Decoder(nn.Module):
#     def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
#         super().__init__()
#         self.output_dim = output_dim + 2  # Add 2 for <sos> and <eos>
#         self.embedding = nn.Embedding(output_dim + 2, emb_dim)  # Add 2 for <sos> and <eos>
#         self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
#         self.fc_out = nn.Linear(hid_dim, output_dim + 2)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, input, hidden, cell):
#         input = input.unsqueeze(0)
#         embedded = self.dropout(self.embedding(input))
#         output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
#         prediction = self.fc_out(output.squeeze(0))
#         return prediction, hidden, cell

# class Seq2Seq(nn.Module):
#     def __init__(self, encoder, decoder, device):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.device = device
        
#     def forward(self, src, trg, teacher_forcing_ratio=0.5):
#         trg_len = trg.shape[0]
#         batch_size = trg.shape[1]
#         trg_vocab_size = self.decoder.output_dim
#         outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
#         hidden, cell = self.encoder(src)
#         input = trg[0,:]
        
#         for t in range(1, trg_len):
#             output, hidden, cell = self.decoder(input, hidden, cell)
#             outputs[t] = output
#             top1 = output.argmax(1)
#             input = trg[t] if random.random() < teacher_forcing_ratio else top1
        
#         return outputs

import torch
import torch.nn as nn
import random
from attention import Attention

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim + 2, emb_dim)  # Add 2 for <sos> and <eos>
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim + 2  # Add 2 for <sos> and <eos>
        self.embedding = nn.Embedding(output_dim + 2, emb_dim)  # Add 2 for <sos> and <eos>
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim + 2)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        
    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden[-1], encoder_outputs)
        a = a.unsqueeze(1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[0,:]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else top1
        
        return outputs
