# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from src.dvqa_param import args
from src.lxrt.entry import LXRTEncoder
from src.lxrt.modeling import BertLayerNorm, GeLU
from src.lxrt.fc import FCNet, GTH

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20

# random number, >=, multiply activations by dropout mask, multiply activations
# by correction (1 / (1 - dropout_rate))
DROPOUT_FLOPS = 4

# compute mean activation (sum), computate variance of activation
# (square and sum), bias (add), scale (multiply)
LAYER_NORM_FLOPS = 5

# GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
ACTIVATION_FLOPS = 8

# max/substract (for stability), exp, sum, divide
SOFTMAX_FLOPS = 5


class VQAModel(nn.Module):
    def __init__(self, num_answers, pooled_layer_norm=False, reweigh_xmodal=False, reweigh_lang=False, reweigh_vision=False):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH,
            reweigh_lang=reweigh_lang,
            reweigh_vision=reweigh_vision
        )
        hid_dim = self.lxrt_encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        if pooled_layer_norm:
            self.pooled_layer_norm = BertLayerNorm(hid_dim, eps=1e-12)

        self.reweigh_xmodal=reweigh_xmodal
        if reweigh_xmodal:
            self.gate_fc = nn.Linear(hid_dim, hid_dim)

    def forward(self, feat, pos, sent, return_feats=False, return_weights=False):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos))

        if self.reweigh_xmodal:
            x_gate = nn.functional.sigmoid(self.gate_fc(x))
            x = x_gate * x

        logit = self.logit_fc(x)

        if return_feats:
            if return_weights:
                return logit, self.pooled_layer_norm(x), x_gate
            else:
                return logit, x
        else:
            if return_weights:
                return logit, x_gate
            else:
                return logit


class CausalVQAModel(nn.Module):
    def __init__(self, num_answers, bias_dim_factor, pooled_layer_norm=False, contrastive=False, tie_training=False):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        self.hid_dim = hid_dim
        self.num_answers = num_answers

        # VQA Answer heads
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
        # assert not (contrastive and tie_training)

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

    def get_flops(self):

        classification_flops = dict(
            hidden_1=2 * self.hid_dim * self.hid_dim * 2,
            hidden_bias=self.hid_dim * 2,
            hidden_act=ACTIVATION_FLOPS * self.hid_dim * 2,
            layer_norm=LAYER_NORM_FLOPS * self.hid_dim * 2,
            logits=2274 * self.hid_dim * 2
        )
        confounder_flops = dict(
            hidden_1=2 * self.hid_dim * self.hid_dim,
            hidden_bias_1=self.hid_dim,
            hidden_act_1=ACTIVATION_FLOPS * self.hid_dim,
            hidden_2=2 * self.hid_dim * int(self.hid_dim/self.bias_dim_factor),
            hidden_bias_2=int(self.hid_dim/self.bias_dim_factor),
            hidden_act_2=ACTIVATION_FLOPS * int(self.hid_dim/self.bias_dim_factor),
            layer_norm=LAYER_NORM_FLOPS * int(self.hid_dim/self.bias_dim_factor),
        )
        fcnet_flops = dict(
            hidden_1=2 * self.hid_dim * self.hid_dim,
            hidden_bias_1=self.hid_dim,
            hidden_act_1=ACTIVATION_FLOPS * self.hid_dim,
            layer_norm=LAYER_NORM_FLOPS * self.hid_dim,
            dropout=DROPOUT_FLOPS * self.hid_dim
        )
        remap_flops = (2 * int(self.hid_dim/self.bias_dim_factor) * self.hid_dim) + (self.hid_dim * 2)

        return sum(classification_flops.values()) + sum(confounder_flops.values()) + remap_flops + sum(fcnet_flops.values())

    def get_ated_flops(self):

        encoder_flops = (768 * 128 * 2) + 128 + (ACTIVATION_FLOPS * 128) + (128 * 64 * 2) + 64
        decoder_flops = (64 * 128 * 2) + 128 + (ACTIVATION_FLOPS * 128) + (128 * 768 * 2) + 768
        norm_flops = 64 * 2
        matmul_flops = 32 * 128 * 10
        attn_flops = 32
        mul_flops = 32 * 768 * 1
        print("Autoencoder only flops", encoder_flops + decoder_flops)
        return encoder_flops + norm_flops + matmul_flops + attn_flops + mul_flops

    def get_dvqa_flops(self):

        fcnet_flops_1_layer = dict(
            hidden_1=2 * self.hid_dim * self.hid_dim,
            hidden_bias_1=self.hid_dim,
            hidden_act_1=ACTIVATION_FLOPS * self.hid_dim,
            layer_norm=LAYER_NORM_FLOPS * self.hid_dim,
            dropout=DROPOUT_FLOPS * self.hid_dim
        )

        cls_flops = dict(
            hidden_1=2 * self.hid_dim * 2274,
            hidden_bias_1=227,
            layer_norm=LAYER_NORM_FLOPS * 2274,
        )

        detect_flops = dict(
            hidden_1=2 * self.hid_dim * 1,
            hidden_bias_1=1,
            layer_norm=LAYER_NORM_FLOPS * 1,
            activation=ACTIVATION_FLOPS * 1
        )

        return (sum(fcnet_flops_1_layer.values())*2 + sum(cls_flops.values()))*3 + sum(detect_flops.values())*3 + SOFTMAX_FLOPS*1 + sum(fcnet_flops_1_layer.values())


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

        #bias_gate = nn.functional.sigmoid(self.bias_gate_fc(bias)) # todo: use biased features to get gate values as well
        #gated_bias = bias_gate*bias
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
            logit = logit-bias_only_logit

        out['logit'] = logit
        out['bias_only_logit'] = bias_only_logit

        return out

        # if return_feats:
        #     return biased_logit, bias_only_logit, self.pooled_layer_norm_bias(bias), self.pooled_layer_norm_debias(x)
        # else:
        #     return biased_logit, bias_only_logit
