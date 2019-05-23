from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.attention import SelfAttention


class DirectInferenceNet(nn.Module):
    # DEPRECATED
    r"""No autoregression."""
    def __init__(self, program_encoder, num_nodes, label_domains, 
                 hidden_size=256, device=None):
        super().__init__()
        self.program_encoder = program_encoder
        self.label_pred_matrix = self.init_pred_layers(
            num_nodes, label_domains, hidden_size)

        self.device = device if device else torch.device('cpu')
        self.hidden_size = hidden_size
        self.label_domains = label_domains
        self.num_nodes = num_nodes

    def init_pred_layers(self, n, doms, dim):
        modules = [
            nn.Linear(10, doms[i])
            for i in range(n)
        ]
        return nn.ModuleList(modules)

    def forward(self, program_args):
        # program_emb = self.program_encoder(
        #     *program_args, return_hiddens=True)
        
        # map from raw input embeddings...
        program_emb = program_args[0][:, 1:-1] - 4
        program_emb = program_emb.float()

        outputs = []
        for i in range(self.num_nodes):
            out_i = self.label_pred_matrix[i](program_emb)
            outputs.append(out_i)

        return outputs


class AutoregressiveInferenceNet(nn.Module):
    r"""Amortized Autoregressive Inference Network for Constrained Probabilistic Grammars.

    @param program_encoder: instance of a ProgramEncoder class.
                            see rnn_encoder.py
    @param num_nodes: integer
                      number of nodes
    @param label_domains: list/numpy array/torch tensor
                          number of categories for each node in ascending global index
                          from 0 to num_nodes - 1
    @param 
                           number of nodes in the continuous embedding layer 
                           for both NODES and ASSIGNMENTS.
    @param hidden_size: integer
                        number of hidden nodes in the (main) RNN.
    @param hidden_dropout: float [default: 0.2]
                           probability of dropping out nodes from the hidden embedding.
    @param batch_norm: boolean [default: False]
                       should we apply a batch norm layer before the label prediction layer?
    @param num_attention_heads: integer [default: 5]
                                number of heads for self_attention
                                https://arxiv.org/abs/1703.03130
    """
    def __init__(self, program_encoder, num_nodes, label_domains, embedding_size=300, 
                 hidden_size=256, hidden_dropout=0.2, num_attention_heads=5, 
                 use_batchnorm=False, device=None):
        super().__init__()
        self.label_embedding_matrix = self.init_embeddings(num_nodes, label_domains, embedding_size)
        self.node_embedding = nn.Embedding(num_nodes + 1, embedding_size)
        self.program_encoder = program_encoder
        self.autoreg_rnn = nn.GRUCell(
            program_encoder.hidden_size + embedding_size * 2,
            hidden_size)
        self.dropout_layer = nn.Dropout(p=hidden_dropout)
        self.autoreg_batchnorm_layer = nn.BatchNorm1d(hidden_size)

        self.self_attention = None
        self.use_attention = num_attention_heads > 0
        if self.use_attention:
            self.self_attention = SelfAttention(
                hidden_size, 
                hidden_size,
                num_attention_heads)
            self.label_pred_matrix = self.init_pred_layers(
                num_nodes, label_domains, hidden_size * num_attention_heads)
            self.label_batchnorm_layer = nn.BatchNorm1d(
                hidden_size * num_attention_heads)
        else:
            self.label_pred_matrix = self.init_pred_layers(
                num_nodes, label_domains, hidden_size)
            self.label_batchnorm_layer = nn.BatchNorm1d(hidden_size)

        self.device = device if device else torch.device('cpu')
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.use_batchnorm = use_batchnorm
        self.label_domains = label_domains
        self.num_nodes = num_nodes
        self.num_attention_heads = num_attention_heads
    
    def init_embeddings(self, n, doms, dim):
        # pad_hack... we want to vectorize things so we learn an extra
        # embedding layer for the pad token...
        modules = [nn.Embedding(doms[i], dim) for i in range(n)]
        modules = modules + [nn.Embedding(2, dim)]
        return nn.ModuleList(modules)

    def init_pred_layers(self, n, doms, dim):
        modules = [nn.Linear(dim, doms[i]) for i in range(n)]
        modules = modules + [nn.Linear(dim, 2)]
        return nn.ModuleList(modules)

    def init_rnn_hiddens(self, batch_size):
        hidden_state = torch.empty(batch_size, self.hidden_size, device=self.device)
        nn.init.xavier_normal_(hidden_state)
        
        return hidden_state

    def step(self, node_i, node_ip1,  program_emb, h0, hidden_store, node_assignments, execution_lengths):
        
        batch_size = node_i.size(0)
        node_i_emb = self.node_embedding(node_i)  # batch_size x embedding_size
                                                    # this represents the node index        

        assign_i = []
        for j in range(batch_size):
            # node_i[j] is a single number representing a global index
            # node_assignments[j, node_i[j].item()] is again a single number
            #   but it picks out the assignment for node_i[j]
            #   we unsqueeze to make a 1x1 tensor

            assign_ij = node_assignments[j, node_i[j].item()].unsqueeze(0)
            # we index the right embedding function T by the global index
            #   assign_ij: 1 x embedding_size
            assign_ij = self.label_embedding_matrix[node_i[j].item()](assign_ij)
            assign_i.append(assign_ij)

        assign_i_emb = torch.cat(assign_i, dim=0)  # batch_size x embedding_size
        # batch_size x (program_encoder.hidden_size + embedding_size * 2)
        input_i_emb = torch.cat((assign_i_emb, program_emb, node_i_emb), dim=1) 
        
        if self.use_batchnorm:
            # add batch normalization on top of h0
            # hopefully for more regularization
            h0 = self.autoreg_batchnorm_layer(h0)        
          
        if self.hidden_dropout > 0:
            # dropout some of the hidden embeddings
            h0_dp = self.dropout_layer(h0)
            h0 = self.autoreg_rnn(input_i_emb, h0_dp)  # hidden_size
        else:
            h0 = self.autoreg_rnn(input_i_emb, h0)          

        hidden_store.append(h0.unsqueeze(1))

        alphas_i = None
        
        if self.use_attention:
            attn_hiddens = torch.cat(hidden_store, dim=1)
            attn_lengths = [max(execution_lengths[j].item(), len(hidden_store))
                                    for j in range(batch_size)]
            attn_lengths = torch.LongTensor(attn_lengths)
            attn_lengths = attn_lengths.to(self.device)
            # NOTE: should we add dropout to attn_hiddens?
            h0_attn, alphas_i = self.self_attention(attn_hiddens, attn_lengths)

        if self.use_batchnorm:
            # add batch normalization on top of last h0 before label_pred_matrix...
            if self.use_attention:
                h0_attn = self.label_batchnorm_layer(h0_attn)
            else:
                h0 = self.label_batchnorm_layer(h0)            
            
        output_i = []
        for j in range(batch_size):
            # label_pred_matrix[node_ip1[j].item()] is a linear layer over
            # the categories of the next node
            if self.use_attention:
                h0_j = h0_attn[j].unsqueeze(0)
            else:
                h0_j = h0[j].unsqueeze(0)
            output_ij = self.label_pred_matrix[node_ip1[j].item()](h0_j)
            output_ij = output_ij.squeeze(0)
            output_i.append(output_ij)        


        return output_i, h0, alphas_i

    def forward(self, execution_trace, execution_lengths, node_assignments, program_args, h0=None):
        r"""
        execution_trace: batch_size by max_trace_length; note this may not equal num_nodes
                         each row looks like: [START,1,5,2,7,...,END,PAD,PAD,...]
                         the number represents the global index of the node.
        execution_lengths: batch_size
                           tells you how many non-pad tokens are in each row of execution_trace.
        node_assignments: batch_size by num_nodes
                          each row looks like: [0,4,1,2,5,7,...,0,0,0]
                          the number represents the assignment of that node.
                          START,PAD,END tokens will always assign to 0.
        program_args: miscellaneous arguments that should be passed to 
                      the program_encoder. See <rnn_encoder.py>.
        outputs: list of list of tensors
                 batch_size by max_trace_length - 1 by num categories
        """
        batch_size = execution_trace.size(0)
        program_emb = self.program_encoder(*program_args, return_hiddens=True)
        max_trace_length = execution_trace.size(1)

        if h0 is None:
            h0 = self.init_rnn_hiddens(batch_size)
            h0 = h0.to(self.device)

        # hack: we need to support computation with padding so add a column of 0s to node_assignments
        #       This will learn a useless embedding for the pad variable
        pad_assignments = torch.zeros(batch_size, 1, device=self.device).long()
        node_assignments = torch.cat((node_assignments, pad_assignments), dim=1)

        hidden_store = []
        # it is important that we loop through max_trace_length in the outer loop
        # this way we can still do the RNN computation in a minibatch. However, this
        # means wasting computation on pad tokens...
        outputs = []
        alphas = []  # store attention weights if possible
        for i in range(max_trace_length - 1):
            node_i = execution_trace[:, i].long()  # batch_size
            # we use h0 to predict the assignment of the next node
            node_ip1 = execution_trace[:, i+1].long()

            # If attention = False, alpha_i will be None
            # This method will modify hidden_store by appending the new h0 to it
            output_i, h0, alphas_i = self.step(node_i, node_ip1, program_emb, h0, hidden_store, node_assignments, execution_lengths)

            if self.use_attention:
                alphas.append(alphas_i)

            outputs.append(output_i)
            

        # outputs is currently max_trace_length - 1 x batch_size x num categories
        # we do not want to take pad into account, so replace them with None
        for j in range(batch_size):
            lengths_j = execution_lengths[j].item() - 1  # -1 bc we ignore last char
            for i in range(lengths_j, max_trace_length - 1):
                outputs[i][j] = None

        # reshape this to batch_size x max_trace_length - 1 x num_categories
        outputs_ = [[outputs[j][i] for j in range(max_trace_length - 1)] for i in range(batch_size)]

        # we keep alphas as size max_trace_length - 1 x batch_size x alpha_matrix
        return outputs_, alphas 


class FeedforwardInferenceNet(nn.Module):
    r"""A baseline to AutoregressiveInferenceNet, where no RNN is used. We directly try 
    to predict things from a single program embedding."""

    def __init__(self, program_encoder, num_nodes, label_domains, 
                 hidden_size=256, hidden_dropout=0.2, device=None, **kwargs):
        super().__init__()
        self.program_encoder = program_encoder
        self.dropout_layer = nn.Dropout(p=hidden_dropout)

        self.feedforward_nn = nn.Linear(program_encoder.hidden_size, hidden_size)

        self.label_pred_matrix = self.init_pred_layers(
            num_nodes, label_domains, hidden_size)

        self.device = device if device else torch.device('cpu')
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.label_domains = label_domains
        self.num_nodes = num_nodes
    
    def init_embeddings(self, n, doms, dim):
        modules = [nn.Embedding(doms[i], dim) for i in range(n)]
        return nn.ModuleList(modules)

    def init_pred_layers(self, n, doms, dim):
        modules = [nn.Linear(dim, doms[i]) for i in range(n)]
        return nn.ModuleList(modules)

    def forward(self, program_args):
        program_emb = self.program_encoder(*program_args, return_hiddens=True)

        if self.hidden_dropout > 0 and self.training:
            program_emb = self.dropout_layer(program_emb)
        
        # a single hidden layer
        program_emb = F.relu(self.feedforward_nn(program_emb))

        # we now project this to every single possible label
        outputs = []
        for i in range(self.num_nodes):
            output_i = self.label_pred_matrix[i](program_emb)
            outputs.append(output_i)

        return outputs  # num_nodes x (categorical_dim for output_i)
