import os, sys
import wandb
import random
import collections
from tqdm import tqdm
from pprint import pprint
from collections import Counter

import json
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader

from src.param import args
import src.config as config

from src.tasks.vqa_model import VQAModel, CausalVQAModel
from src.lxrt.optimization import BertAdam
from src.tasks.losses import FocalLoss, Plain, RateDistortionUnconstrained, ContrastiveLoss
from src.pretrain.qa_answer_table import load_lxmert_qa
from src.tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator

# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def farm_batch_sampler(batch_size, labels):

    dataset_len = len(labels)
    class_size = max(labels) + 1
    batch_idxs = []

    for class_idx in range(0, class_size):
        sample_idxs = [i for i in range(dataset_len) if labels[i] == class_idx]
        random.shuffle(sample_idxs)
        for j in range(0, len(sample_idxs), batch_size):
            batch_idxs.append(sample_idxs[j:j+batch_size])

    random.shuffle(batch_idxs)
    return batch_idxs


def get_data_tuple(splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:
    dset = VQADataset(splits)
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)

    if 'train' in splits and args.use_farm:
        # get batch sampler as per category
        question_types = set([e['question_type'] for e in dset.entries])
        qtype2idx = {qtype: i for i, qtype in enumerate(question_types)}
        qtype_labels = [qtype2idx[e['question_type']] for e in dset.entries]
        sampler_by_type = farm_batch_sampler(bs, qtype_labels)

        data_loader = DataLoader(
            tset, num_workers=args.num_workers, pin_memory=True, batch_sampler=sampler_by_type
        )

    else:
        data_loader = DataLoader(
            tset, batch_size=bs,
            shuffle=shuffle, num_workers=args.num_workers,
            drop_last=drop_last, pin_memory=True
        )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VQA:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=False)
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=1024, shuffle=False, drop_last=False)
        else:
            self.valid_tuple = None
        
        # Model
        if args.causal_model:
            print("Bias dim factor: %s", args.bias_dim_factor)
            self.model = CausalVQAModel(self.train_tuple.dataset.num_answers, bias_dim_factor=args.bias_dim_factor,
                                        pooled_layer_norm=args.use_farm, contrastive=args.contrastive, tie_training=args.tie_training)

        else:
            self.model = VQAModel(self.train_tuple.dataset.num_answers, pooled_layer_norm=args.use_farm, reweigh_xmodal=args.reweigh_xmodal, reweigh_lang=args.reweigh_lang, reweigh_vision=args.reweigh_vision)

        if args.freeze:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if 'logit' in name or 'gate' in name:
                        pass
                    else:
                        param.requires_grad = False
        

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
        if args.loss_fn == 'Plain':
            print("Initializing cross entropy based loss function")
            self.loss_fn = Plain()
        elif args.loss_fn == 'Focal':
            print("Initializing focal loss based loss function")
            self.loss_fn = FocalLoss()
        elif args.loss_fn == 'Farm':
            print("Initializing rate distortion based loss function")
            self.loss_fn = RateDistortionUnconstrained()
        else:
            raise RuntimeError('not implement for {}'.format(args.loss_fn))

        if args.contrastive:
            self.contrastive_loss = ContrastiveLoss(1.0)

        if args.loss_fn != "Farm" and args.display_farm:
            self.farm_loss = RateDistortionUnconstrained()

        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=args.warmup_factor,
                                  t_total=t_total)
        else:
            #self.optim = args.optimizer(self.model.parameters(), args.lr)
            self.optim = args.optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)
        if args.save_logit:
            os.makedirs(os.path.join(self.output, 'logit'), exist_ok=True)

    def train(self, train_tuple, eval_tuple):

        if args.causal_model:
            pdist = nn.PairwiseDistance(p=2)

        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            if epoch >= 5:
                break
            torch.cuda.empty_cache()
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target, mask, score) in iter_wrapper(enumerate(loader)):
                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                weights = None
                if args.use_farm:
                    if args.causal_model:
                        # logit, bias_only_logit, bias_only_feats, feats = self.model(feats, boxes, sent, return_feats=True)
                        out = self.model(feats, boxes, sent, return_feats=True)
                        assert out['logit'].dim() == target.dim() == 2
                    else:
                        if args.reweigh_xmodal:
                            logit, feats, weights = self.model(feats, boxes, sent, return_feats=True, return_weights=args.reweigh_xmodal)
                        else:
                            logit, feats = self.model(feats, boxes, sent, return_feats=True)
                        assert logit.dim() == target.dim() == 2
                else:
                    if args.causal_model:
                        # logit, bias_only_logit, bias_only_feats, feats = self.model(feats, boxes, sent, return_feats=True)
                        out = self.model(feats, boxes, sent, return_feats=True)
                        assert out['logit'].dim() == target.dim() == 2
                    elif args.reweigh_xmodal:
                        logit, weights = self.model(feats, boxes, sent, return_weights=args.reweigh_xmodal)
                    else:
                        logit = self.model(feats, boxes, sent)
                        assert logit.dim() == target.dim() == 2

                dict_args = {}
                if config.use_miu:
                    dict_args['miu'] = score.cuda()
                    dict_args['mask'] = mask.cuda()

                if args.use_farm:
                    type_labels = torch.tensor(np.zeros(args.batch_size, dtype=np.int8))
                    if args.causal_model:
                        bias_only_rate_loss, bias_only_ce_loss = self.loss_fn(out['bias'], type_labels, out['bias_only_logit'], target, out['bias'].device)
                        rate_loss, ce_loss = self.loss_fn(out['feature'], type_labels, out['logit'], target, feats.device)
                        # distance_loss = -torch.mean(pdist(bias_feats, debias_feats))
                        rate_loss = -rate_loss
                        bias_only_rate_loss = -bias_only_rate_loss
                        logit = out['logit']
                        if args.contrastive:
                            contrastive_loss = self.contrastive_loss(out['feature'], out['debiased_feature'], out['bias'])
                        if i %100 == 0:
                            if args.wandb:
                                wandb.log({"rate_loss_bias": bias_only_rate_loss.item(),
                                    "rate_loss": abs(rate_loss.item()),
                                    "ce_loss": ce_loss.item(),
                                    "ce_loss_bias": bias_only_ce_loss.item()})
                                    # "distance_loss": distance_loss.item()}
                                if args.contrastive:
                                    wandb.log({"contrastive_loss": contrastive_loss.item()})
                            print("Train Epoch %s, Step %s: Rate distortion (biased/bias_only) = %s / %s, Cross-entropy (biased/bias_only) = %s / %s" % (epoch, i, round(rate_loss.item(), 2), round(bias_only_rate_loss.item(), 2), round(ce_loss.item(), 2), round(bias_only_ce_loss.item(), 2)))
                            if args.contrastive:
                                print("Contrastive Loss = %s" % contrastive_loss.item())
                    else:
                        rate_loss, ce_loss = self.loss_fn(feats, type_labels, logit, target, feats.device)
                        if i %100 == 0:
                            if args.wandb:
                                wandb.log({"rate_loss_debias": rate_loss.item(),
                                    "ce_loss": ce_loss.item()})
                                if args.reweigh_xmodal:
                                    wandb.log({"gate_mean": torch.mean(weights.detach()).item(),
                                        "gate_std": torch.std(weights.detach()).item()})
                            try:
                                print("Train Epoch %s, Step %s: Rate distortion loss = %s, Cross-entropy loss = %s" % (epoch, i, round(rate_loss.item(), 2), round(ce_loss.item(), 2)))
                            except:
                                pass
                    if epoch >= 0:
                        if args.dynamic_coeff:
                            if args.causal_model:
                                farm_coeff = float(min(1.0, abs(args.farm_coeff / (
                                            bias_only_rate_loss.item() / bias_only_ce_loss.item()))))

                                # loss = rate_loss * farm_coeff + ce_loss + biased_ce_loss + distance_loss
                                if epoch < args.bias_epochs:
                                    loss = bias_only_rate_loss * farm_coeff + ce_loss + bias_only_ce_loss
                                else:
                                    loss = ce_loss
                                if args.contrastive:
                                    loss = loss + contrastive_loss
                            else:
                                farm_coeff = float(min(1.0, abs(args.farm_coeff / (rate_loss.item() / ce_loss.item()))))
                                loss = rate_loss * farm_coeff + ce_loss
                            if i%100 == 0:
                                print("Farm coefficient = %s" % round(farm_coeff, 2))
                        else:
                            loss = rate_loss*args.farm_coeff + ce_loss
                    else:
                        loss = ce_loss
                else:
                    if args.causal_model:
                        bias_only_ce_loss = self.loss_fn(out['bias_only_logit'], target, **dict_args)
                        ce_loss = self.loss_fn(out['logit'], target, **dict_args)
                        loss = bias_only_ce_loss + ce_loss
                        if args.contrastive:
                            contrastive_loss = self.contrastive_loss(out['feature'], out['debiased_feature'],
                                                                     out['bias'])
                            loss = loss + contrastive_loss
                        if args.display_farm:
                            type_labels = torch.tensor(np.zeros(args.batch_size, dtype=np.int8))
                            with torch.no_grad():
                                bias_only_rate_loss, _ = self.farm_loss(out['bias'], type_labels, out['bias_only_logit'], target, out['bias'].device)
                                rate_loss, _ = self.farm_loss(out['feature'], type_labels, out['logit'], target, feats.device)
                            # distance_loss = -torch.mean(pdist(bias_feats, debias_feats))
                            rate_loss = -rate_loss
                            bias_only_rate_loss = -bias_only_rate_loss

                        logit = out['logit']
                        if i %100 == 0:
                            print("Train Epoch %s, Step %s: Rate distortion (biased/bias_only) = %s / %s, Cross-entropy (biased/bias_only) = %s / %s" % (epoch, i, round(rate_loss.item(), 2), round(bias_only_rate_loss.item(), 2), round(ce_loss.item(), 2), round(bias_only_ce_loss.item(), 2)))
                            if args.contrastive:
                                print("Contrastive Loss = %s" % contrastive_loss.item())
                            if args.wandb:
                                wandb.log({"rate_loss_bias": bias_only_rate_loss.item(),
                                    "rate_loss": abs(rate_loss.item()),
                                    "ce_loss": ce_loss.item(),
                                    "ce_loss_bias": bias_only_ce_loss.item()})
                                    # "distance_loss": distance_loss.item()}
                                if args.contrastive:
                                    wandb.log({"contrastive_loss": contrastive_loss.item()})

                    else:
                        loss = self.loss_fn(logit, target, **dict_args)
                        if i%100 == 0:
                            if args.wandb:
                                wandb.log({"ce_loss": loss.item()})
                                if args.reweigh_xmodal:
                                     wandb.log({"gate_mean": torch.mean(weights.detach()).item(),
                                          "gate_std": torch.std(weights.detach()).item()})

                        print("Train Epoch %s, Step %s: loss = %s" % (epoch, i, round(loss.item(), 2)))


                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, ans in zip(ques_id, label.cpu().numpy()):
                    quesid2ans[qid.item()] = ans

            print("\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans)[0] * 100.))

            if self.valid_tuple is not None:  # Do Validation
                # valid_score = self.evaluate(eval_tuple)
                valid_score, save_results = self.evaluate(eval_tuple, os.path.join(args.output, 'val_predict_epoch_%s.json' % epoch), epoch)
                if valid_score > best_valid:

                    best_results = save_results
                    best_valid = valid_score
                    self.save(args.name.replace('.pth', '_%s.pth' % epoch), epoch, best_valid)

                print("Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) +
                      "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.))

                if args.wandb:
                    wandb.log({"train_accuracy": evaluator.evaluate(quesid2ans)[0] * 100.,
                               "eval_accuracy": valid_score * 100})

        import json
        cp = 'CP' if config.cp_data else 'NC'
        miu = 'miu' if config.use_miu else 'base'
        with open('./result_{}_{}_{}.json'.format(
                cp, config.version, miu), 'w') as fd:
            json.dump(best_results, fd)

    def predict(self, eval_tuple: DataTuple, dump=None, epoch=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {} #needs to be changed to dictionary for normal inference
        quesid2ans_bias = {}
        quesid2ans_tie = {}
        features = None

        for i, datum_tuple in tqdm(enumerate(loader), total=len(loader)):
            ques_id, feats, boxes, sent, target = datum_tuple[:5]
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                if args.causal_model:
                    out = self.model(feats, boxes, sent)

                    score, label = out['logit'].max(1)
                    for qid, l in zip(ques_id, label.cpu().numpy()):
                        ans = dset.label2ans[l]
                        quesid2ans[qid.item()] = ans
                    score, label = out['bias_only_logit'].max(1)
                    for qid, l in zip(ques_id, label.cpu().numpy()):
                        ans = dset.label2ans[l]
                        quesid2ans_bias[qid.item()] = ans

                    if args.tie_inference:
                        tie_logit = out['logit'] - out['bias_only_logit']
                        score, label = tie_logit.max(1)
                        for qid, l in zip(ques_id, label.cpu().numpy()):
                            ans = dset.label2ans[l]
                            quesid2ans_tie[qid.item()] = ans

                    # if args.save_logit:
                    #     if epoch is None:
                    #         epoch = -1
                    #     torch.save(out['logit'].detach().cpu(), os.path.join(self.output, 'logit', 'biased_%s_%s.pt' % (epoch, i)))
                    #     torch.save(out['bias_only_logit'].detach().cpu(), os.path.join(self.output, 'logit', 'bias_%s_%s.pt' % (epoch, i)))

                else:
                    # logit = self.model(feats, boxes, sent)
                    logit, feats = self.model(feats, boxes, sent, return_feats=True)
                    if features is None:
                        features = feats.detach().cpu().numpy()
                    else:
                        features = np.concatenate((features, feats.detach().cpu().numpy()), axis=0)
                    print(features.shape)
                    score, label = logit.max(1)
                    confs = torch.nn.functional.softmax(logit, dim=1)
                    for k, (qid, l) in enumerate(zip(ques_id, label.cpu().numpy())):
                        ans = dset.label2ans[l]
                        # quesid2ans[qid.item()] = ans
                        # quesid2ans[qid.item()] = {"answer": ans, "conf": confs[k][l].item()}
                        quesid2ans.append({"qid": qid.item(), "answer": ans, "conf": confs[k][l].item()})
                    if i > 0 and i%100 == 0:
                        np.save(dump.replace('.json', '_%s.npy' % int(i/100)), features)
                        features = None


        if dump is not None:
            print("Saving at %s" % dump)
            evaluator.dump_result(quesid2ans, dump)
            acc, results = evaluator.evaluate(quesid2ans)
            with open(dump.replace('.json', '_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            #evaluator.dump_results(quesid2ans, dump) #uncomment previous lines for normal results
            # Dump features
            np.save(dump.replace('.json', '.npy'), features)
            if args.causal_model:
                print("Bias-only accuracy", evaluator.evaluate(quesid2ans_bias)[0])
                print("Bias-only answer distribution", Counter(list(quesid2ans_bias.values())))
                evaluator.dump_result(quesid2ans_bias, dump.replace('.json', '_bias.json'))
                if args.tie_inference:
                    print("TIE inference accuracy", evaluator.evaluate(quesid2ans_tie)[0])
                    evaluator.dump_result(quesid2ans_tie, dump.replace('.json', '_tie.json'))

        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None, epoch=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump, epoch)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    def save(self, name, epoch, best_val_score):
        results = {
            'epoch': epoch + 1,
            'best_val_score': best_val_score,
            'model_state': self.model.state_dict(),
            'optim_state': self.optim.state_dict(),
            'loss_state': self.loss_fn.state_dict(),
        }
        torch.save(results, os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        # TODO: fix for finetuning
        print("Load model from %s" % path)
        state_dict = torch.load("%s" % path)
        try:
            self.model.load_state_dict(state_dict['model_state'], strict=False)
        except KeyError:
            self.model.load_state_dict(state_dict, strict=True)
        #self.optim.load_state_dict(state_dict['optim_state'])
        #self.loss_fn.load_state_dict(state_dict['loss_state'])


if __name__ == "__main__":
    print(args)
    print_keys = ['cp_data', 'version', 'use_miu']
    print_dict = {key: getattr(config, key) for key in print_keys}
    pprint(print_dict, width=150)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = False


    if args.wandb:
        wandb.init(project="vqacp", entity="adyasha", config={"seed": args.seed})

    # Build Class
    vqa = VQA()

    # print("Additional FLOPS", vqa.model.get_flops())
    # print("Additional FLOPS in DQA", vqa.model.get_dvqa_flops())
    # print("Additional FLOPS in ATE-D", vqa.model.get_ated_flops())
    # sys.exit()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # Test or Train
    if args.test is not None:
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(args.test, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple('val', bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'val_predict.json')
            )
        elif 'train' in args.test:
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple('train', bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'train_predict.json')
            )
        else:
            assert False, "No such test option for %s" % args.test
    else:
        vqa.train(vqa.train_tuple, vqa.valid_tuple)
