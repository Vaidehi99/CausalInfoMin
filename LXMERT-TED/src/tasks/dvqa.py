# coding=utf-8
# Copyleft 2019 project LXRT.
import wandb
import collections
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch._six import string_classes

from src.dvqa_param import args
print("imported args")
from src.pretrain.qa_answer_table import load_lxmert_qa
from src.tasks.losses import RateDistortionUnconstrained
from vqa_model import VQAModel, CausalVQAModel
# from model import VQAModel_probe
import os
import psutil
from torch.utils.data.dataloader import default_collate
process = psutil.Process(os.getpid())
import re
import torch.nn.functional as F
import numpy as np
from collections import Counter

@torch.no_grad()
def evaluate(model, dataloader, args):

    evaluate_bias = args.causal_model

    print("Evaluate bias is ", evaluate_bias)
    score = 0
    bias_score = 0
    bias_result = {}
    upper_bound = 0
    num_data = 0
    entropy = 0
    for i, (v, b, q, a, q_id) in tqdm(enumerate(dataloader)):
        v = v.cuda()
        b = b.cuda()
        q = list(q)
        q_id = q_id.cuda()
        out_dict = model(v, b, q)
        pred = out_dict['logit']
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score.item()

        if evaluate_bias:
            bias_pred = out_dict['bias_only_logit']
            bias_batch_score = compute_score_with_logits(bias_pred, a.cuda()).sum()
            bias_score += bias_batch_score.item()
            for i in range(bias_pred.size(0)):
                bias_result[q_id[i].item()] = get_answer(bias_pred[i], dataloader)

        upper_bound += (a.max(1)[0]).sum().item()
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    if evaluate_bias:
        print("Bias-only accuracy", bias_score / len(dataloader.dataset))
        print("Bias-only answer distribution", Counter(list(bias_result.values())))

        if args.wandb:
            wandb.log({"bias_accuracy": bias_score / len(dataloader.dataset)})

    return score, upper_bound


def trim_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    _use_shared_memory = True
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        if 1 < batch[0].dim(): # image features
            max_num_boxes = max([x.size(0) for x in batch])
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = len(batch) * max_num_boxes * batch[0].size(-1)
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            # warning: F.pad returns Variable!
            return torch.stack([F.pad(x, (0,0,0,max_num_boxes-x.size(0))).data for x in batch], 0, out=out)
        else:
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [trim_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class Contrastive_loss(nn.Module):
    def __init__(self, tao=1):
        super(Contrastive_loss, self).__init__()
        self.sim = nn.CosineSimilarity(dim=-1)
        self.tao = tao

    def forward(self, fea, pos_fea, neg_fea):
        fea = F.normalize(fea, dim=1)
        pos_fea = F.normalize(pos_fea, dim=1)
        neg_fea = F.normalize(neg_fea, dim=1)

        pos_sim = self.sim(fea, pos_fea)
        neg_sim = self.sim(fea, neg_fea)

        logits = torch.exp(pos_sim / self.tao) / \
            (torch.exp(pos_sim / self.tao) + torch.exp(neg_sim / self.tao))
        loss = (-1.0 * torch.log(logits))

        return loss.mean()

# multi-label soft loss
def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def compute_self_loss(logits_neg, a):
    prediction_ans_k, top_ans_ind = torch.topk(
        F.softmax(a, dim=-1), k=1, dim=-1, sorted=False)
    neg_top_k = torch.gather(
        F.softmax(logits_neg, dim=-1), 1, top_ans_ind).sum(1)

    qice_loss = neg_top_k.mean()
    return qice_loss


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


def get_our_data():
    from dvqa_data import Dictionary, VQAFeatureDataset
    from src.dvqa_param import args as opt

    dictionary = Dictionary.load_from_file(os.path.join(opt.dataroot, 'dictionary.pkl'))
    opt.ntokens = dictionary.ntoken

    train_dset = VQAFeatureDataset('train', dictionary, opt.dataroot, opt.img_root, ratio=opt.ratio, adaptive=False)  # load labeld data
    eval_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root,ratio=1.0, adaptive=False)

    train_loader = DataLoader(train_dset, opt.batch_size, shuffle=True, num_workers=4, collate_fn=trim_collate)
    opt.use_all = 1
    val_loader = DataLoader(eval_dset, opt.batch_size, shuffle=False, num_workers=4, collate_fn=trim_collate)
    return train_loader, val_loader

class VQA:
    def __init__(self,folder="/",load=True):

        # Datasets
        self.train_loader, self.val_loader = get_our_data()
        if args.causal_model:
            print("Bias dim factor: %s", args.bias_dim_factor)
            self.model = CausalVQAModel(2274, bias_dim_factor=args.bias_dim_factor,
                                        pooled_layer_norm=args.use_farm, contrastive=args.contrastive, tie_training=args.tie_training)
        else:
            self.model = VQAModel(2274)
        #Change
        # self.model = VQAModel(2410)
        # self.model = VQAModel(2)

        # self.model = VQAModel_probe(2)
        #Change

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        if args.loss_fn == 'Farm':
            print("Initializing rate distortion based loss function")
            self.loss_fn = RateDistortionUnconstrained()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
        self.contrastive_loss = Contrastive_loss(tao=1)

        if load :
            if 'bert' in args.optim:
                batch_per_epoch = len(self.train_loader)
                t_total = int(batch_per_epoch * args.epochs)
                print("BertAdam Total Iters: %d" % t_total)
                from src.lxrt.optimization import BertAdam
                self.optim = BertAdam(list(self.model.parameters()),
                                      lr=args.lr,
                                      warmup=0.1,
                                      t_total=t_total)
            else:
                self.optim = args.optimizer(self.model.parameters(), args.lr)
            # Output Directory
            self.output = args.output
            os.makedirs(self.output, exist_ok=True)

    def train_causal(self, train_loader, val_loader):
        best_valid = 0.
        # print(args.epochs)
        # print("Arg epochs")
        # exit()

        # train
        total_num = len(train_loader.dataset)
        for epoch in range(args.epochs):

            torch.cuda.empty_cache()

            total_loss = 0
            self_sup = epoch >= args.pretrain_epoches
            for i, (feats, boxes, sent, target, ques_id) in tqdm(enumerate(train_loader)):

                self.model.train()
                self.optim.zero_grad()
                batch_size = feats.size(0)

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                out = self.model(feats, boxes, list(sent), return_feats=True)

                if args.use_farm:
                    type_labels = torch.tensor(np.zeros(batch_size, dtype=np.int8))
                    bias_only_rate_loss, bias_only_ce_loss = self.loss_fn(out['bias'], type_labels, out['bias_only_logit'], target, out['bias'].device)
                    rate_loss, ce_loss = self.loss_fn(out['feature'], type_labels, out['logit'], target, feats.device)
                    rate_loss = -rate_loss
                    bias_only_rate_loss = -bias_only_rate_loss
                    if args.dynamic_coeff:
                        farm_coeff = float(min(1.0, abs(args.farm_coeff / (
                                    bias_only_rate_loss.item() / bias_only_ce_loss.item()))))
                    else:
                        farm_coeff = float(min(1.0, abs(args.farm_coeff / (rate_loss.item() / ce_loss.item()))))
                    loss = bias_only_rate_loss * farm_coeff + ce_loss + bias_only_ce_loss
                    if args.contrastive:
                        con_loss = self.contrastive_loss(out['feature'], out['debiased_feature'], out['bias'])
                        loss = loss + con_loss
                else:
                    loss = instance_bce_with_logits(
                        out['logits'], target, reduction='mean')

                total_loss += loss.item()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score_pos = compute_score_with_logits(out['logit'], target.data).sum()
                total_loss += loss.item() * batch_size

                if i and i % 100 == 0:
                    log_str = 'traing: %d/%d, train_loss: %.6f, rate_loss (debias/bias): %.6f / %.6f, ce_loss (debias/bias): %.6f / %.6f, constrast_loss: %.6f, train_acc: %.6f' % (
                    i, len(train_loader), total_loss / total_num,
                    rate_loss, bias_only_rate_loss, ce_loss, bias_only_ce_loss, con_loss, score_pos)
                    print(log_str)

                    if args.wandb:
                        wandb.log({"rate_loss_bias": bias_only_rate_loss.item(),
                                   "rate_loss": abs(rate_loss.item()),
                                   "ce_loss": ce_loss.item(),
                                   "ce_loss_bias": bias_only_ce_loss.item()})
                        # "distance_loss": distance_loss.item()}
                        if args.contrastive:
                            wandb.log({"contrastive_loss": con_loss.item()})

            self.save("LAST")

            # if self.valid_tuple is not None:  # Do Validation
            self.model.train(False)
            valid_score, upper_bound = evaluate(self.model, val_loader, args)
            self.model.train(True)
            if valid_score > best_valid:
                best_valid = valid_score
                self.save("BEST")

            log_str = "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                      "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str)

            if args.wandb:
                wandb.log({"train_accuracy": score_pos.item(),
                           "eval_accuracy": valid_score * 100})

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

            if epoch == 50:
                break

        return best_valid

    def train(self, train_loader, val_loader):
        best_valid = 0.
        # print(args.epochs)
        # print("Arg epochs")
        # exit()

        # train
        total_num = len(train_loader.dataset)
        for epoch in range(args.epochs):
            total_loss = 0
            total_bce_loss = 0
            self_loss = 0
            total_self_loss = 0
            train_score_pos = 0
            train_score_neg_q = 0
            train_score_neg_v = 0
            total_q_bce_loss = 0
            total_v_bce_loss = 0
            total_debias_bce_loss = 0
            total_con_loss = 0
            self_sup = epoch>= args.pretrain_epoches
            for i, (feats, boxes, sent, target, ques_id) in tqdm(enumerate(train_loader)):

                self.model.train()
                self.optim.zero_grad()
                batch_size = feats.size(0)

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                out_dict = self.model(feats, boxes, list(sent), self_sup)

                # base VQA model
                #Change
                # print(out_dict['logits'].shape)
                # print(target.shape)
                # exit()
                #Change
                bce_loss = instance_bce_with_logits(
                    out_dict['logits'], target, reduction='mean')
                # only_q
                bce_loss_q = instance_bce_with_logits(
                    out_dict['q_logits'], target, reduction='mean')
                # only_v
                bce_loss_v = instance_bce_with_logits(
                    out_dict['v_logits'], target, reduction='mean')
                # debias
                bce_loss_debias = instance_bce_with_logits(
                    out_dict['debias_logits'], target, reduction='mean')
                con_loss = self.contrastive_loss(
                    out_dict['fea'], out_dict['pos_fea'], out_dict['neg_fea'])

                loss = bce_loss + bce_loss_q + bce_loss_debias + \
                    bce_loss_v + con_loss

                if self_sup:
                    self_loss_q = compute_self_loss(out_dict['logit_neg_q'], target)
                    self_loss_v = compute_self_loss(out_dict['logit_neg_v'], target)

                    self_loss = self_loss_v + args.self_loss_q * self_loss_q
                    loss = loss + args.self_loss_weight * self_loss

                total_loss += loss.item()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score_pos = compute_score_with_logits(out_dict['logits'], target.data).sum()
                train_score_pos += score_pos.item()
                total_loss += loss.item() * batch_size
                total_bce_loss += bce_loss.item() * batch_size
                total_con_loss += con_loss.item() * batch_size
                total_q_bce_loss += bce_loss_q.item() * batch_size
                total_debias_bce_loss += bce_loss_debias.item() * batch_size
                total_v_bce_loss += bce_loss_v.item() * batch_size

                if self_sup:
                    score_neg_q = compute_score_with_logits(
                    out_dict['logit_neg_q'], target.data).sum()
                    score_neg_v = compute_score_with_logits(
                        out_dict['logit_neg_v'], target.data).sum()
                    total_self_loss += self_loss.item() * batch_size
                    train_score_neg_q += score_neg_q.item()
                    train_score_neg_v += score_neg_v.item()
                if i and i%100==0:
                    log_str = 'traing: %d/%d, train_loss: %.6f, bce_loss: %.6f, q_bce_loss: %.6f, v_bce_loss: %.6f, debias_bce_loss: %.6f, constrast_loss: %.6f, self_loss: %.6f, neg_train_q_acc: %.6f, neg_train_v_acc: %.6f, pos_train_acc: %.6f' %(i, len(train_loader), total_loss / total_num,
                     total_bce_loss /total_num, total_q_bce_loss / total_num, total_v_bce_loss / total_num,
                     total_debias_bce_loss /
                     total_num, total_con_loss /
                     total_num, total_self_loss / total_num,
                     100 * train_score_neg_q / total_num, 100 * train_score_neg_v / total_num, 100 * train_score_pos / total_num)          
                    print(log_str)

            self.save("LAST")
            
            # if self.valid_tuple is not None:  # Do Validation
            self.model.train(False)
            valid_score, upper_bound = evaluate(self.model, val_loader)
            self.model.train(True)
            if valid_score > best_valid:
                best_valid = valid_score
                self.save("BEST")

            log_str = "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                        "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str)

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()
            
            if epoch == 50:
                break

        return best_valid

    def train_probe(self, train_loader, val_loader):
        best_valid = 0.

        for param in self.model.lxrt_encoder.parameters():
                    param.requires_grad = False

        print("Parameters being trained")
        for param in self.model.parameters():
                    if param.requires_grad:
                        print(param.shape)

        # train
        total_num = len(train_loader.dataset)
        for epoch in range(args.epochs):
            total_loss = 0
            total_bce_loss = 0
            self_loss = 0
            total_self_loss = 0
            train_score_pos = 0
            train_score_neg_q = 0
            train_score_neg_v = 0
            total_q_bce_loss = 0
            total_v_bce_loss = 0
            total_debias_bce_loss = 0
            total_con_loss = 0
            for i, (feats, boxes, sent, target, ques_id) in tqdm(enumerate(train_loader)):

                self.model.train()

                for param in self.model.lxrt_encoder.parameters():
                    if(param.requires_grad):
                        print("Sth wrong")
                
                self.optim.zero_grad()
                batch_size = feats.size(0)

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                out_dict = self.model(feats, boxes, list(sent), False)

                # base VQA model
                #Change
                # print(out_dict['logits'].shape)
                # print(target.shape)
                # exit()
                #Change
                bce_loss = instance_bce_with_logits(
                    out_dict['logits'], target, reduction='mean')
                

                loss = bce_loss 

               
                total_loss += loss.item()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score_pos = compute_score_with_logits(out_dict['logits'], target.data).sum()
                train_score_pos += score_pos.item()
                total_loss += loss.item() * batch_size
                total_bce_loss += bce_loss.item() * batch_size
                

                if i and i%100==0:
                    log_str = 'traing: %d/%d, train_loss: %.6f, bce_loss: %.6f, q_bce_loss: %.6f, v_bce_loss: %.6f, debias_bce_loss: %.6f, constrast_loss: %.6f, self_loss: %.6f, neg_train_q_acc: %.6f, neg_train_v_acc: %.6f, pos_train_acc: %.6f' %(i, len(train_loader), total_loss / total_num,
                     total_bce_loss /total_num, total_q_bce_loss / total_num, total_v_bce_loss / total_num,
                     total_debias_bce_loss /
                     total_num, total_con_loss /
                     total_num, total_self_loss / total_num,
                     100 * train_score_neg_q / total_num, 100 * train_score_neg_v / total_num, 100 * train_score_pos / total_num)          
                    print(log_str)

            self.save("LAST")
            
            # if self.valid_tuple is not None:  # Do Validation
            self.model.train(False)
            valid_score, upper_bound = evaluate(self.model, val_loader)
            self.model.train(True)
            if valid_score > best_valid:
                best_valid = valid_score
                self.save("BEST")

            log_str = "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                        "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str)

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()
            
            if epoch == 49:
                break

        return best_valid

    def save(self, name):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model_to_save.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict, strict=False)

        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        print("Fully loaded model")

    def load_dvqa(self, path):

        print("Load pre-trained model DVQA from %s" % path)
        state_dict = torch.load(path)

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            elif key.startswith("lxrt_encoder.model.module."):
                new_state_dict[key[len("lxrt_encoder.model.module."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)

    def load_other(self, path):
        print("Load model partially from %s" % path)
        state_dict = torch.load("%s" % path)
        # self.optim.load_state_dict(state_dict['optim_state'])
        # self.loss_fn.load_state_dict(state_dict['loss_state'])
        own_state = self.model.state_dict()
        # print(state_dict.keys())
        # exit()
        # print(own_state.keys())
        # exit()
        for name, param in state_dict['model_state'].items():
            if name.startswith('lxrt_encoder.model.'):
                name = name.replace('lxrt_encoder.model.','lxrt_encoder.model.module.')
            # print(name)
            # exit()
            if name not in own_state:
                 print(name)
                 continue
            #if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
            else:
                param = param.data
            # print(name)
            # print(name in state_dict.keys())
            own_state[name].copy_(param)

    def load_some(self, path):
        print("Load model partially from %s" % path)
        state_dict = torch.load("%s" % path)
        #self.optim.load_state_dict(state_dict['optim_state'])
        #self.loss_fn.load_state_dict(state_dict['loss_state'])
        own_state = self.model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 print(name)
                 continue
            #if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
            else:
                param = param.data
            # print(name)
            # print(name in state_dict.keys())
            own_state[name].copy_(param)

#For getting language-image features
image_dataroot="/playpen/vaidehi/D-VQA/data/trainval" 


if __name__ == "__main__":

    print(args)

    vqa = VQA()
    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.

    if args.load is not None:
        # print(loading)
        vqa.load(args.load)
    # for param in vqa.model.lxrt_encoder.parameters():
    #     param.requires_grad=False
    # print("requiring grad") 
    # for param in vqa.model.lxrt_encoder.model.module.bert.encoder.AE_vision.parameters():
    #     param.requires_grad = False
            # print(param.shape)
    # exit()

    if args.loadDVQA is not None:
        vqa.load_dvqa(args.loadDVQA)

    if args.wandb:
        wandb.init(project="vqacp", entity="adyasha", config={"seed": args.seed})

    vqa.train_causal(vqa.train_loader, vqa.val_loader)
    # vqa.train_probe(vqa.train_loader, vqa.val_loader)
