"""
Transformer model.  (c) 2021 Georgia Tech

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

import numpy as np

import torch
from torch import nn
import random

####### Do not modify these imports.

class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        
        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the embedding lookup.          #
        ##############################################################################
        self.embeddingL = nn.Embedding(input_size, hidden_dim)
        self.posembeddingL = nn.Embedding(max_length, hidden_dim)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        
        ##############################################################################
        # Deliverable 2: Initializations for multi-head self-attention.              #
        # You don't need to do anything here. Do not modify this code.               #
        ##############################################################################
        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the feed-forward layer.        # 
        ##############################################################################
        
        self.ff_linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.ff_relu = nn.ReLU()
        self.ff_linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################
        
        self.final_linear = nn.Linear(hidden_dim, output_size)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.
        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """

        #############################################################################
        # TODO:
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #
        #############################################################################

        embeddings = self.embed(inputs)
        attn_out = self.multi_head_attention(embeddings)
        ff_out = self.feedforward_layer(attn_out)
        outputs = self.final_layer(ff_out)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        #############################################################################
        # TODO:
        # Deliverable 1: Return the embeddings.                                     #
        #############################################################################
      
        N, T = inputs.shape
        positions = torch.arange(T, device=inputs.device).unsqueeze(0).expand(N, T)
        embeddings = self.embeddingL(inputs) + self.posembeddingL(positions)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        #############################################################################

        # Head 1
        K1 = self.k1(inputs)   # (N, T, dim_k)
        V1 = self.v1(inputs)   # (N, T, dim_v)
        Q1 = self.q1(inputs)   # (N, T, dim_q)
        score1 = self.softmax(torch.bmm(Q1, K1.transpose(1, 2)) / (self.dim_k ** 0.5))
        head1 = torch.bmm(score1, V1)   # (N, T, dim_v)

        # Head 2
        K2 = self.k2(inputs)
        V2 = self.v2(inputs)
        Q2 = self.q2(inputs)
        score2 = self.softmax(torch.bmm(Q2, K2.transpose(1, 2)) / (self.dim_k ** 0.5))
        head2 = torch.bmm(score2, V2)

        # Concatenate heads and project back to hidden_dim
        multi_head = torch.cat([head1, head2], dim=2)   # (N, T, 2*dim_v)
        projected = self.attention_head_projection(multi_head)

        # Residual connection + layer norm
        outputs = self.norm_mh(inputs + projected)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        #############################################################################

        ff = self.ff_linear2(self.ff_relu(self.ff_linear1(inputs)))
        outputs = self.norm_ff(inputs + ff)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        #############################################################################

        outputs = self.final_linear(inputs)
                
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        

class FullTransformerTranslator(nn.Module):

    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2,
                 dim_feedforward=2048, num_layers_enc=2, num_layers_dec=2, dropout=0.2, max_length=43, ignore_index=1):
        super(FullTransformerTranslator, self).__init__()

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.pad_idx=ignore_index

        seed_torch(0)

        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the Transformer Layer          #
        ##############################################################################

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers_enc,
            num_decoder_layers=num_layers_dec,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        ##############################################################################
        # TODO:
        # Deliverable 2: Initialize what you need for the embedding lookup.          #
        ##############################################################################
        # Do not change the order for these variables
        self.srcembeddingL = nn.Embedding(input_size, hidden_dim)
        self.tgtembeddingL = nn.Embedding(output_size, hidden_dim)
        self.srcposembeddingL = nn.Embedding(max_length, hidden_dim)
        self.tgtposembeddingL = nn.Embedding(max_length, hidden_dim)

        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the final layer.               #
        ##############################################################################

        self.final_linear = nn.Linear(hidden_dim, output_size)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, src, tgt):
        """
         :param src: a PyTorch tensor of shape (N,T)
                tgt: a PyTorch tensor of shape (N,T)
         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the full Transformer stack for the forward pass. #
        #############################################################################
        outputs=None
        # shift tgt to right, add one <sos> to the beginning and shift the other tokens to right
        tgt = self.add_start_token(tgt)

        N, T = src.shape

        # embed src and tgt
        src_pos = torch.arange(T, device=src.device).unsqueeze(0).expand(N, T)
        tgt_pos = torch.arange(T, device=tgt.device).unsqueeze(0).expand(N, T)
        src_emb = self.srcembeddingL(src) + self.srcposembeddingL(src_pos)
        tgt_emb = self.tgtembeddingL(tgt) + self.tgtposembeddingL(tgt_pos)

        # causal mask for decoder self-attention
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=src.device)
        tgt_key_padding_mask = (tgt == self.pad_idx)

        transformer_out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        outputs = self.final_linear(transformer_out)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def generate_translation(self, src):
        """
         :param src: a PyTorch tensor of shape (N,T)
         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # TODO:
        # Deliverable 5: Generate translation autoregressively.                     #
        #############################################################################
        N, T = src.shape
        outputs = torch.zeros(N, T, self.output_size).to(self.device)
        # start with all-pad target, first token is <sos>=2
        tgt = torch.full((N, T), self.pad_idx, dtype=torch.long, device=self.device)
        tgt[:, 0] = 2   # <sos>

        for t in range(T):
            out = self.forward(src, tgt)        # (N, T, output_size)
            outputs[:, t, :] = out[:, t, :]
            if t + 1 < T:
                tgt[:, t + 1] = out[:, t, :].argmax(dim=1)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def add_start_token(self, batch_sequences, start_token=2):
        def has_consecutive_start_tokens(tensor, start_token):
            consecutive_start_tokens = torch.tensor([start_token, start_token], dtype=tensor.dtype,
                                                    device=tensor.device)
            is_consecutive_start_tokens = torch.all(tensor[:, :2] == consecutive_start_tokens, dim=1)
            return torch.all(is_consecutive_start_tokens).item()

        if has_consecutive_start_tokens(batch_sequences, start_token):
            return batch_sequences

        modified_sequences = batch_sequences.clone()
        start_token_tensor = torch.tensor(start_token, dtype=modified_sequences.dtype, device=modified_sequences.device)
        start_token_tensor = start_token_tensor.view(1, -1)
        modified_sequences[:, 1:] = batch_sequences[:, :-1]
        modified_sequences[:, 0] = start_token_tensor
        return modified_sequences

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True