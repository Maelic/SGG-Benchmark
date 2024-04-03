"""
Based on the implementation of https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""
import torch
import torch.nn as nn
from sgg_benchmark.modeling.roi_heads.relation_head.models.model_transformer import MultiHeadAttention, PositionwiseFeedForward

class Single_Att_Layer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(Single_Att_Layer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, q_input, k_input, v_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            q_input, k_input, v_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask.float()

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask.float()

        return enc_output, enc_slf_attn

class Self_Attention_Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.transformer_layer = Single_Att_Layer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)

    def forward(self, input_feats, num_objs):

        input_feats = input_feats.split(num_objs, dim=0)
        input_feats = nn.utils.rnn.pad_sequence(input_feats, batch_first=True)

        # -- Prepare masks
        bsz = len(num_objs)
        device = input_feats.device
        pad_len = max(num_objs)
        num_objs_ = torch.LongTensor(num_objs).to(device).unsqueeze(1).expand(-1, pad_len)
        slf_attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(num_objs_).unsqueeze(1).expand(-1, pad_len, -1) # (bsz, pad_len, pad_len)
        non_pad_mask = torch.arange(pad_len, device=device).to(device).view(1, -1).expand(bsz, -1).lt(num_objs_).unsqueeze(-1) # (bsz, pad_len, 1)

        # -- Forward
        enc_output, enc_slf_attn = self.transformer_layer(
                input_feats, input_feats, input_feats,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        enc_output = enc_output[non_pad_mask.squeeze(-1)]
        return enc_output

class Cross_Attention_Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.transformer_layer = Single_Att_Layer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)

    def forward(self, visual_feats, textual_feats, num_objs):

        visual_feats = visual_feats.split(num_objs, dim=0)
        visual_feats = nn.utils.rnn.pad_sequence(visual_feats, batch_first=True)
        textual_feats = textual_feats.split(num_objs, dim=0)
        textual_feats = nn.utils.rnn.pad_sequence(textual_feats, batch_first=True)

        # -- Prepare masks
        bsz = len(num_objs)
        device = visual_feats.device
        pad_len = max(num_objs)
        num_objs_ = torch.LongTensor(num_objs).to(device).unsqueeze(1).expand(-1, pad_len)
        slf_attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(num_objs_).unsqueeze(1).expand(-1, pad_len, -1) # (bsz, pad_len, pad_len)
        non_pad_mask = torch.arange(pad_len, device=device).to(device).view(1, -1).expand(bsz, -1).lt(num_objs_).unsqueeze(-1) # (bsz, pad_len, 1)

        # -- Forward
        enc_output, enc_slf_attn = self.transformer_layer(
                visual_feats, textual_feats, textual_feats,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        enc_output = enc_output[non_pad_mask.squeeze(-1)]
        return enc_output