from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from src.models.transformer import TransformerEncoder
from globals import PAD_IDX, UNK_IDX, START_IDX, END_IDX

class MegaProgramEncoder(nn.Module):
    r"""Contains 2 TokenCharProgramEncoders -- one for anonymized code and one for raw code."""
    def __init__(self, token_vocab_size, char_vocab_size, anon_token_vocab_size, anon_char_vocab_size, 
                 output_size, max_seq_len, char_max_seq_len, anon_max_seq_len, anon_char_max_seq_len,
                 model_type='rnn', learnable_alpha=True, hidden_size=256, word_dropout=0, 
                 device=None, **encoder_kwargs):
        super().__init__()
        self.raw_encoder = TokenCharProgramEncoder( token_vocab_size, char_vocab_size, output_size, 
                                                    max_seq_len, char_max_seq_len, word_dropout=word_dropout, 
                                                    learnable_alpha=learnable_alpha, hidden_size=hidden_size,
                                                    model_type=model_type, device=device, **encoder_kwargs)
        self.anon_encoder = TokenCharProgramEncoder(anon_token_vocab_size, anon_char_vocab_size, output_size, 
                                                    anon_max_seq_len, anon_char_max_seq_len, word_dropout=word_dropout, 
                                                    learnable_alpha=learnable_alpha, hidden_size=hidden_size,
                                                    model_type=model_type, device=device, **encoder_kwargs)
        if learnable_alpha:
            self.alpha_fc = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid())
        else:
            self.alpha = 0.5

        self.learnable_alpha = learnable_alpha
        self.output_fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.word_dropout = word_dropout

    def forward(self, token_sequence, token_lengths, char_sequence, char_lengths, 
                anon_token_sequence, anon_token_lengths, anon_char_sequence, anon_char_lengths, 
                return_hiddens=False):
        raw_hiddens = self.raw_encoder(token_sequence, token_lengths, char_sequence, char_lengths, return_hiddens=True)
        anon_hiddens = self.anon_encoder(anon_token_sequence, anon_token_lengths, anon_char_sequence, anon_char_lengths, 
                                         return_hiddens=True)
        if self.learnable_alpha:
            concated = torch.cat((raw_hiddens, anon_hiddens), dim=1)
            alpha = self.alpha_fc(concated)
        else:
            alpha = self.alpha
        weighted = raw_hiddens * alpha + anon_hiddens * (1 - alpha)

        if return_hiddens:
            return weighted
        
        return self.output_fc(weighted)


class TokenCharProgramEncoder(nn.Module):
    def __init__(self, token_vocab_size, char_vocab_size, output_size, 
                 max_seq_len, char_max_seq_len, model_type='rnn', 
                 learnable_alpha=True, word_dropout=0, hidden_size=256, 
                 device=None, **encoder_kwargs):
        super().__init__()
        self.token_encoder = ProgramEncoder(token_vocab_size, output_size, max_seq_len, 
                                            model_type=model_type, hidden_size=hidden_size, 
                                            word_dropout=word_dropout, device=device, **encoder_kwargs)
        self.char_encoder = ProgramEncoder( char_vocab_size, output_size, char_max_seq_len, 
                                            model_type=model_type, hidden_size=hidden_size, 
                                            word_dropout=word_dropout, device=device, **encoder_kwargs)

        if learnable_alpha:
            self.alpha_fc = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid())
        else:
            self.alpha = 0.5

        self.learnable_alpha = learnable_alpha
        self.output_fc = nn.Linear(hidden_size, output_size)
        self.word_dropout = word_dropout
        self.hidden_size = hidden_size
        

    def forward(self, token_sequence, token_lengths, char_sequence, char_lengths, return_hiddens=False):
        token = self.token_encoder(token_sequence, token_lengths, return_hiddens=True)
        char = self.char_encoder(char_sequence, char_lengths, return_hiddens=True)
        if self.learnable_alpha:
            concated = torch.cat((token, char), dim=1)
            alpha = self.alpha_fc(concated)
        else:
            alpha = self.alpha
        weighted = token * alpha + char * (1 - alpha)

        if return_hiddens:
            return weighted
        
        return self.output_fc(weighted)


class RawAnonProgramEncoder(nn.Module):
    def __init__(self, token_vocab_size, anon_token_vocab_size, output_size, 
                 max_seq_len, anon_max_seq_len, model_type='rnn', 
                 learnable_alpha=True, word_dropout=0, hidden_size=256, 
                 device=None, **encoder_kwargs):
        super().__init__()
        self.raw_encoder = ProgramEncoder(token_vocab_size, output_size, max_seq_len, 
                                          model_type=model_type, hidden_size=hidden_size,
                                          word_dropout=word_dropout, device=device, **encoder_kwargs)
        self.anon_encoder = ProgramEncoder(anon_token_vocab_size, output_size, anon_max_seq_len, 
                                           model_type=model_type, hidden_size=hidden_size,
                                           word_dropout=word_dropout, device=device, **encoder_kwargs)

        if learnable_alpha:
            self.alpha_fc = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid())
        else:
            self.alpha = 0.5

        self.learnable_alpha = learnable_alpha
        self.output_fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.word_dropout = word_dropout

    def forward(self, token_sequence, token_lengths, anon_token_sequence, anon_token_lengths, return_hiddens=False):
        raw_hiddens = self.raw_encoder(token_sequence, token_lengths, return_hiddens=True)
        anon_hiddens = self.anon_encoder(anon_token_sequence, anon_token_lengths, return_hiddens=True)
        if self.learnable_alpha:
            concated = torch.cat((raw_hiddens, anon_hiddens), dim=1)
            alpha = self.alpha_fc(concated)
        else:
            alpha = self.alpha
        weighted = raw_hiddens * alpha + anon_hiddens * (1 - alpha)

        if return_hiddens:
            return weighted
        
        return self.output_fc(weighted)


class ProgramEncoder(nn.Module):
    def __init__(self, vocab_size, output_size, max_seq_len, model_type='rnn', 
                 word_dropout=0, hidden_size=256, device=None, **encoder_kwargs):
        super().__init__()
        assert model_type in ['rnn', 'transformer']
        if model_type == 'rnn':
            self.encoder = RNNProgramEncoder(vocab_size, output_size, word_dropout=word_dropout,
                                             hidden_size=hidden_size, device=device, **encoder_kwargs)
        else:
            self.encoder = TransformerProgramEncoder(vocab_size, output_size, max_seq_len, 
                                                     word_dropout=word_dropout, hidden_size=hidden_size,
                                                     device=device, **encoder_kwargs)
        self.output_fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.word_dropout = word_dropout

    def forward(self, input_sequence, seq_lengths, return_hiddens=False):
        hiddens = self.encoder(input_sequence, seq_lengths, return_hiddens=True)

        if return_hiddens:
            return hiddens
        
        output = self.output_fc(hiddens)

        return output


class RNNProgramEncoder(nn.Module):
    # TODO: add bidirectional?
    def __init__(self, vocab_size, output_size, hidden_size=256, device=None, 
                 # RNN specific hyperparameters
                 word_dropout=0, embedding_size=300, cell_type="gru", num_layers=1, hidden_dropout=0):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell_type = cell_type.lower()
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.word_dropout = word_dropout
        self.hidden_dropout = hidden_dropout
        self.device = device if device else torch.device('cpu')

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        kwargs = {
            "input_size": self.embedding_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.hidden_dropout,
            "batch_first": True
        }

        if self.cell_type == "rnn":
            self.rnn = nn.RNN(**kwargs)
        elif self.cell_type == "lstm":
            self.rnn = nn.LSTM(**kwargs)
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(**kwargs)
        else:
            raise ValueError("cell_type must be one of rnn, lstm gru. Got [{}]".format(cell_type))

        self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.to(device=device)

    def forward(self, input_sequence, seq_lengths, h0=None, return_hiddens=False):
        # Shape:  n x max_seq_len
        batch_size = input_sequence.size(0)

        if batch_size > 1:
            # sort in decreasing order of length in order to pack
            # sequence; if only 1 element in batch, nothing to do.
            seq_lengths, sorted_idx = torch.sort(seq_lengths, descending=True)
            input_sequence = input_sequence[sorted_idx]

        if self.word_dropout > 0 and self.training:
            input_sequence = dropout_words(input_sequence, self.word_dropout)

        # Shape:  n x max_seq_len x embedding_size
        input_embedding = self.embedding(input_sequence)

        packed_input = rnn_utils.pack_padded_sequence(
            input_embedding,
            seq_lengths.data.tolist(),
            batch_first=True
        )

        if h0 is None:
            h0 = self.init_hidden(batch_size)

        # _, hidden = self.rnn(packed_input, None)
        # n x max_seq_len x embedding_size
        if self.cell_type == 'lstm':
            _, (rnn_hidden, _) = self.rnn(packed_input, h0)
        else:
            _, rnn_hidden = self.rnn(packed_input, h0)

        # rnn_output, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)

        # Shape: (n x num_layers x output_size)
        rnn_hidden = rnn_hidden.permute(1, 0, 2).contiguous()
        rnn_hidden = rnn_hidden[:, -1, :]   # Take only last layer

        if batch_size > 1:
            # reverse the sorting used to pack padded seq
            _, reversed_idx = torch.sort(sorted_idx)
            rnn_hidden = rnn_hidden[reversed_idx]

        if return_hiddens:
            return rnn_hidden

        output = self.fc(rnn_hidden)

        return output

    def init_hidden(self, batch_size):
        hidden_state = torch.empty(self.num_layers, batch_size, self.hidden_size, device=self.device)
        nn.init.xavier_normal_(hidden_state)

        if self.cell_type == "lstm":
            cell_state = torch.empty(self.num_layers, batch_size, self.hidden_size, device=self.device)
            nn.init.xavier_normal_(cell_state)
            return cell_state, hidden_state
        else:
            return hidden_state


class TransformerProgramEncoder(nn.Module):
    def __init__(self, vocab_size, output_size, max_seq_len, hidden_size=256, device=None, 
                 # transformer specific hyperparams
                 word_dropout=0, n_layers=6, n_head=8, d_k=64, d_v=64, d_inner=2048, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.word_dropout = word_dropout
        self.device = device if device else torch.device('cpu')

        self.transformer = TransformerEncoder(vocab_size, max_seq_len, 
                                              d_word_vec=self.hidden_size,
                                              d_model=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.to(device=device)
    
    def forward(self, input_sequence, seq_lengths, return_hiddens=False):
        if self.word_dropout > 0 and self.training:
            input_sequence = dropout_words(input_sequence, self.word_dropout)

        rnn_hidden, _ = self.transformer(input_sequence, seq_lengths)

        if return_hiddens:
            return rnn_hidden
        
        output = self.fc(rnn_hidden)

        return output


def dropout_words(seq, dropout_proba, sos_idx=START_IDX, eos_idx=END_IDX, 
                  pad_idx=PAD_IDX, unk_idx=UNK_IDX):
    # randomly replace with unknown tokens
    prob = torch.rand(seq.size())
    # don't dropout important tokens by forcing
    # their keep probability to be 1
    prob[((seq.cpu().data == sos_idx) | \
          (seq.cpu().data == eos_idx) | \
          (seq.cpu().data == pad_idx))] = 1

    mask_seq = seq.clone()
    mask_seq[(prob < dropout_proba).to(seq.device)] = unk_idx
    
    return mask_seq


# Test code to ensure shapes match
if __name__ == '__main__':
    import torch

    def make_stub_data():
        stub_x = torch.tensor([
            [2, 5, 0, 0, 3],
            [2, 5, 9, 0, 3]
        ])
        lens = torch.tensor([2, 3])
        return stub_x, lens

    num_labels = 4
    vocab_size = 10
    hidden_size = 100
    output_size = num_labels
    model = RNNProgramEncoder(vocab_size, output_size)

    X, lensX = make_stub_data()
    y = model(X, lensX)

    print(y.size())
    assert(y.size() == torch.Size((2, 4)))
