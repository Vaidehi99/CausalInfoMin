# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
import torch
from src.param import args
from src.lxrt.entry import LXRTEncoder
from src.lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20


class VQAModel_original(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
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

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x, deconf_loss = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)

        return logit, deconf_loss

subs_conf_all = torch.load("/ssd-playpen/home/vaidehi/vaidehi/Deconfounded_disbiasing/cluster_centroids_language_sub_conf_AE_lang_lxmert_only.pt")
subs_conf_all = subs_conf_all.cuda()
class VQAModel_inference(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        ).cuda()
        hid_dim = self.lxrt_encoder.dim
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        ).cuda()
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
        # x, deconf_loss = self.lxrt_encoder(sent, (feat, pos), subs_conf)
        # print(next(self.lxrt_encoder.parameters()).is_cuda)
        # print([x.shape for x in subs_conf_all])
        # exit()
        # print(subs_conf_all[0].shape)
        # print(type(self.lxrt_encoder(subs_conf_all[0].expand(len(sent),-1,-1), sent, (feat, pos))))
        
        # print(subs_conf_all.shape)
        # print(len(sent))
        # print(feat.shape)
        # y = subs_conf_all[0].expand(len(sent),-1,-1)
        # print(y[0,2,61])
        # print(y[106,2,61])
        # print(y[949,2,61])
        # print(y.shape)
        x_all = torch.stack([self.lxrt_encoder(subs_conf_all[i].expand(len(sent),-1,-1), sent, (feat, pos))[0] for i in range(subs_conf_all.shape[0])]).cuda()
        # print(x_all.shape)
        # exit()
        # x_all = torch.stack([self.lxrt_encoder(subs_conf, sent, (feat, pos)) for subs_conf in subs_conf_all[:2]]).cuda()
        # logit = self.logit_fc(x)
        # print(self.logit_fc(x_all[0]).shape)
        print("infernece")
        logits_all = torch.stack([self.logit_fc(x) for x in x_all]).cuda()
        logit = torch.mean(logits_all, dim=0)
        # print(logit.shape)
        # exit()

        return logit, 0.0


