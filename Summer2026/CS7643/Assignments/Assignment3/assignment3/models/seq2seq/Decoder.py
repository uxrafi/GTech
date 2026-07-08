"""
S2S Decoder model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN", attention=False):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type
        self.attention = attention

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer based on the "model_type" argument.            #
        #          Supported types (strings): "RNN", "LSTM". Instantiate the        #
        #          appropriate layer for the specified model_type.                  #
        #          Set batch_first to true.                                         #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #       5) If attention is True, A linear layer to downsize concatenation   #
        #           of context vector and input                                     #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################

        self.embedding = nn.Embedding(output_size, emb_size)

        if model_type == "RNN":
            self.rnn = nn.RNN(emb_size, decoder_hidden_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(emb_size, decoder_hidden_size, batch_first=True)

        self.linear = nn.Linear(decoder_hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

        if attention:
            self.attention_linear = nn.Linear(emb_size + encoder_hidden_size, emb_size)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def compute_attention(self, hidden, encoder_outputs):
        """ compute attention probabilities given a controller state (hidden) and encoder_outputs using cosine similarity
            as your attention function.

                cosine similarity (q,K) =  q@K.Transpose / |q||K|
                hint |K| has dimensions: N, T
                Where N is batch size, T is sequence length

            Args:
                hidden (tensor): the controller state (dimensions: 1,N, hidden_dim)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention (dimensions: N,T, hidden dim)
            Returns:
                attention: attention probabilities (dimension: N,1,T)
        """

        #############################################################################
        #                              BEGIN YOUR CODE                              #
        #############################################################################

        # hidden: (1, N, hidden_dim) -> (N, 1, hidden_dim)
        q = hidden.permute(1, 0, 2)

        # dot product: (N, 1, T)
        dot = torch.bmm(q, encoder_outputs.transpose(1, 2))

        # norms: q is (N,1,1), K is (N,1,T)
        norm_q = torch.norm(q, dim=2, keepdim=True)
        norm_k = torch.norm(encoder_outputs, dim=2, keepdim=True).permute(0, 2, 1)

        attention_prob = torch.softmax(dot / (norm_q * norm_k + 1e-8), dim=2)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return attention_prob

    def forward(self, input, hidden, encoder_outputs=None):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden state of the previous time step from the decoder, dimensions: (1,N,decoder_hidden_size)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention, dimensions: (N,T,encoder_hidden_size)
                attention (Boolean): If True, need to implement attention functionality
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
                hidden (tensor): the state coming out of the hidden unit, dimensions: (1,N,decoder_hidden_size)
                where N is the batch size, T is the sequence length
        """

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #############################################################################

        # (N, 1, emb_size)
        embedded = self.dropout(self.embedding(input))

        if self.attention and encoder_outputs is not None:
            # attention is only on the hidden state (not cell state) for LSTM
            h = hidden[0] if self.model_type == "LSTM" else hidden
            attn = self.compute_attention(h, encoder_outputs)          # (N, 1, T)
            context = torch.bmm(attn, encoder_outputs)                 # (N, 1, encoder_hidden_size)
            rnn_input = self.attention_linear(
                torch.cat([context, embedded], dim=2)                  # (N, 1, emb+enc_hidden)
            )                                                           # (N, 1, emb_size)
        else:
            rnn_input = embedded

        rnn_out, hidden = self.rnn(rnn_input, hidden)
        output = self.logsoftmax(self.linear(rnn_out.squeeze(1)))      # (N, output_size)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden