import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """ Implementation of self-attention described by Lin et al. 2017
        https://arxiv.org/abs/1703.03130
        Looked at following resources while implementing:
            - https://medium.com/apache-mxnet/sentiment-analysis-via-self-attention-with-mxnet-gluon-dc774d38ba69
            - https://github.com/ExplorerFreda/Structured-Self-Attentive-Sentence-Embedding/blob/master/models.py
    """
    def __init__(self, i_dim, d_dim, r_dim):
        """
        @param i_dim (int): input dimmension
        @param d_dim (int): intermediate dimension (d_a in the paper)
        @param r_dim (int): number of attention hops (r in the paper)
        """
        super(SelfAttention, self).__init__()
        self.i_dim = i_dim
        self.d_dim = d_dim
        self.r_dim = r_dim

        # Initialize Layers
        self.ws1 = nn.Linear(i_dim, d_dim, bias=False)
        self.ws2 = nn.Linear(d_dim, r_dim, bias=False)

    def _gen_att_masks(self, alphas, batch_lengths):
        """ Generate masks for attention probabilities. 
        @param alphas (torch.Tensor): tensor of shape (b, r_dim, l)
        @param batch_lengths (List[int]): List of actual lengths of input sequences.
        @returns att_masks (torch;Tensor): Tensor of sentence masks of shape (b, r_dim, l),
            where b = batch size, r_dim = number of attention hops, and l is the max
            length of an input sequence in the batch.
        """
        att_masks = torch.zeros(alphas.size(), dtype=torch.float)
        for i, src_len in enumerate(batch_lengths):
            att_masks[i, :, src_len:] = 1
        return att_masks.to(self.ws1.weight.device)

    def forward(self, x, batch_lengths):
        """
        @param x (torch.Tensor): input text tensor (b, l, i_dim) where b is batch size,
            l is length of the largest input sequence, i_dim is the input dimension
        @param batch_lengths (torch.Tensor): true lengths of inputs in x
        @returns x_rep (torch.Tensor): representation of input x after applying self attention
            (b, r_dim * i_dim) where b is batch size, r_dim is the number of attention hops, and
            l is the length of the largest input sequence
        """
        size = x.size()  # [b, l, i_dim]
        x_compressed = x.contiguous().view(-1, size[2])  # [b*l, i_dim]

        h_bar = torch.tanh(self.ws1(x_compressed))  # [b*l, d_dim]
        alphas = self.ws2(h_bar).view(size[0], size[1], -1)  # [b, l, r_dim]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [b, r_dim, l]

        masks = self._gen_att_masks(alphas, batch_lengths)
        alphas.data.masked_fill_(masks.byte(), -float('inf'))  # [b, r_dim, l]
        alphas = F.softmax(alphas, dim=2)  # [b, r_dim, l]
        
        x_rep = torch.bmm(alphas, x)
        return x_rep.view(size[0], -1), alphas
