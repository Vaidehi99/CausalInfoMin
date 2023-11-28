# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_GQA_LENGTH = 20


class GQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_GQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)

        return logit


class CausalGQAModel(nn.Module):
    def __init__(self, num_answers, bias_dim_factor, pooled_layer_norm=False, contrastive=False, tie_training=False):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_GQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        self.hid_dim = hid_dim
        self.num_answers = num_answers
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

        self.pooled_layer_norm_debias = BertLayerNorm(hid_dim, eps=1e-12)

        #self.bias_gate_fc = nn.Linear(hid_dim, hid_dim)

        self.bias_dim_factor = bias_dim_factor
        self.confounder_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            GeLU(),
            nn.Linear(hid_dim, int(hid_dim/self.bias_dim_factor)),
            GeLU(),
            BertLayerNorm(int(hid_dim/self.bias_dim_factor), eps=1e-12)
        )

        self.contrastive = contrastive
        self.tie_training = tie_training

        if self.contrastive:

            self.remap_bias = nn.Linear(int(hid_dim/self.bias_dim_factor), hid_dim)

            self.confounder_logit_fc = nn.Sequential(
                nn.Linear(hid_dim, hid_dim * 2),
                GeLU(),
                BertLayerNorm(hid_dim * 2, eps=1e-12),
                nn.Linear(hid_dim * 2, num_answers)
            )
            activation = 'ReLU'
            norm = 'weight'
            self.debias_only = FCNet([hid_dim, hid_dim], norm=norm, dropout=0, act=activation)
            self.pooled_layer_norm_bias = BertLayerNorm(hid_dim, eps=1e-12)
        else:
            self.pooled_layer_norm_bias = BertLayerNorm(int(hid_dim / self.bias_dim_factor), eps=1e-12)
            self.confounder_logit_fc = nn.Sequential(
                nn.Linear(int(hid_dim/self.bias_dim_factor), int(hid_dim/self.bias_dim_factor) * 2),
                GeLU(),
                BertLayerNorm(int(hid_dim/self.bias_dim_factor) * 2, eps=1e-12),
                nn.Linear(int(hid_dim/self.bias_dim_factor) * 2, num_answers)
            )

    def forward(self, feat, pos, sent, return_feats=False):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos))

        x_input = x.detach().clone()
        # get confounder
        bias = self.confounder_fc(x_input)

        # bias_gate = nn.functional.sigmoid(self.bias_gate_fc(bias)) # todo: use biased features to get gate values as well
        # gated_bias = bias_gate*bias
        # debiased_feat = x - bias

        # backup Dec 23
        # bias_only_logit = self.confounder_logit_fc(bias)
        # biased_logit = self.logit_fc(x)
        #
        # if return_feats:
        #     return biased_logit, bias_only_logit, self.pooled_layer_norm_bias(bias), self.pooled_layer_norm_debias(x)
        # else:
        #     return biased_logit, bias_only_logit

        out = {}
        if self.contrastive:
            remapped_bias = self.remap_bias(bias)
            debiased_feat = x - remapped_bias
            debiased_feat = self.debias_only(debiased_feat)
            logit = self.logit_fc(debiased_feat)
            out['feature'] = x
            # out['bias'] = self.pooled_layer_norm_bias(remapped_bias)
            out['bias'] = remapped_bias
            out['debiased_feature'] = debiased_feat
            bias_only_logit = self.confounder_logit_fc(remapped_bias)

        else:

            bias_only_logit = self.confounder_logit_fc(bias)
            logit = self.logit_fc(x)

            out['bias'] = self.pooled_layer_norm_bias(bias)
            out['feature'] = x

        if self.tie_training:
            logit = logit - bias_only_logit

        out['logit'] = logit
        out['bias_only_logit'] = bias_only_logit

        return out