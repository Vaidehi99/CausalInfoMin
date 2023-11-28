# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections
import json
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import numpy as np
import wandb
from src.param import args
from src.pretrain.qa_answer_table import load_lxmert_qa
from src.tasks.gqa_model import GQAModel, CausalGQAModel
from src.tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator
from src.tasks.losses import FocalLoss, Plain, RateDistortionUnconstrained, ContrastiveLoss
from collections import Counter
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = GQADataset(splits)
    tset = GQATorchDataset(dset)
    evaluator = GQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class GQA:
    def __init__(self):
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 512
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        if args.causal_model:
            print("Bias dim factor: %s", args.bias_dim_factor)
            self.model = CausalGQAModel(self.train_tuple.dataset.num_answers,
                                        bias_dim_factor=args.bias_dim_factor,
                                        pooled_layer_norm=args.use_farm, contrastive=args.contrastive, tie_training=args.tie_training)

        else:
            self.model = GQAModel(self.train_tuple.dataset.num_answers)

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

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)

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
            print("Total Iters: %d" % t_total)
            from src.lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            torch.cuda.empty_cache()
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

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

                # logit = self.model(feats, boxes, sent)
                # assert logit.dim() == target.dim() == 2

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


                # if args.mce_loss:
                #     max_value, target = target.max(1)
                #     loss = self.mce_loss(logit, target) * logit.size(1)
                # else:
                #     loss = self.bce_loss(logit, target)
                #     loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

                if args.wandb:
                    wandb.log({"train_accuracy": evaluator.evaluate(quesid2ans)[0] * 100.,
                               "eval_accuracy": valid_score * 100})

            print(log_str, end='')



            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        quesid2ans_bias = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
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

                else:
                    logit = self.model(feats, boxes, sent)
                    score, label = logit.max(1)
                    for qid, l in zip(ques_id, label.cpu().numpy()):
                        ans = dset.label2ans[l]
                        quesid2ans[qid] = ans

        if dump is not None:
            print("Saving at %s" % dump)
            evaluator.dump_result(quesid2ans, dump)

            if args.causal_model:
                print("Bias-only accuracy", evaluator.evaluate(quesid2ans_bias))
                print("Bias-only answer distribution", Counter(list(quesid2ans_bias.values())))
                evaluator.dump_result(quesid2ans_bias, dump.replace('.json', '_bias.json'))

        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)

'''
#For getting language-image features
image_dataroot="/playpen2/home/vaidehi/LXMERT-VQACP/data/vg_gqa_imgfeat/trainval_vg_out" 

def get_img_features_from_id(img_id):
    true_feature_id = ids[int(img_id)]
    feature = m.fetchone(colletion_id=true_feature_id, object_id=1)
    features = torch.from_numpy(np.frombuffer(feature, dtype=np.float32).reshape(2048, 36)).permute(1, 0)
    box = m.fetchone(colletion_id=true_feature_id, object_id=0) 
    boxes = torch.from_numpy(np.frombuffer(box, dtype=np.float32).reshape(4, 36)).permute(1, 0) 
    return features, boxes
   

if __name__ == "__main__":
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    gqa = GQA()
    if args.load is not None:
        gqa.load(args.load)
    gqa.model.eval()
    # print(torch.cuda.current_device())
    # print(torch.cuda.device_count())
    # print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    with torch.no_grad():
    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    #Change
        
    # vqa.to(device)
    

        batch_siz = 1024
        data1 = json.load(open('/playpen2/home/vaidehi/LXMERT-VQACP/data/gqa/train.json'))
        data2 = json.load(open('/playpen2/home/vaidehi/LXMERT-VQACP/data/gqa/valid.json'))
        data = data1 + data2
        print("Training data has {} samples".format(len(data)))
        x_embs_l = torch.zeros(len(data),1,768)
        x_embs_v = torch.zeros(len(data),1,768)
        img = [data[i]['img_id'] for i in range(len(data))]
        questions = [data[i]['sent'] for i in range(len(data))]
        m = MIO(image_dataroot)
        ids = {}
        for i in range(m.size):
            id_= struct.unpack("<I", m.get_collection_metadata(i))[0]
            ids[id_] = i

        # img_batch = torch.stack(get_img_features_from_id(id) for id in img[i:i+batch_size])
        # vqa.model.lxrt_encoder(ques_batch, (torch.randn(len(ques_batch),36,2048), torch.randn(len(ques_batch),36,4)))

        for i in range(0,len(data),batch_siz):
              ques_batch =  questions[i:i+batch_siz]
              feat_list = [get_img_features_from_id(id) for id in img[i:i+batch_siz]]
              img_batch_feat, img_batch_box = torch.stack([feat_list[j][0] for j in range(len(feat_list))]), torch.stack([feat_list[j][1] for j in range(len(feat_list))])
              img_batch_feat, img_batch_box = img_batch_feat.cuda(), img_batch_box.cuda()
              torch.manual_seed(1)
            #   print(vqa.model.lxrt_encoder(ques_batch, (img_batch_feat, img_batch_box))[0][0].shape)
            #   print(len(gqa.model.lxrt_encoder(ques_batch, (img_batch_feat, img_batch_box))[0]))
              (x_lang_feats, x_img_feats), _ = gqa.model.lxrt_encoder(ques_batch, (img_batch_feat, img_batch_box))
            #   print(lang_feats)
            #   print(lang_feats.shape)
            #   exit()
            #   print(torch.mean(lang_feats,1).cpu().shape)
              x_embs_l[i:i+1024,0,:] = torch.mean(x_lang_feats,1).cpu()
              x_embs_v[i:i+1024,0,:] = torch.mean(x_img_feats,1).cpu()
        torch.save(x_embs_l, '/playpen2/home/vaidehi/Deconfounded_disbiasing/cross_language_outputs_gqa_lxrt_orig.pt')
        torch.save(x_embs_v, '/playpen2/home/vaidehi/Deconfounded_disbiasing/cross_vision_outputs_gqa_lxrt_orig.pt')
'''
'''
#For getting image features
image_dataroot="/playpen2/home/vaidehi/LXMERT-VQACP/data/vg_gqa_imgfeat/trainval_vg_out" 

def get_img_features_from_id(img_id):
    true_feature_id = ids[int(img_id)]
    feature = m.fetchone(colletion_id=true_feature_id, object_id=1)
    features = torch.from_numpy(np.frombuffer(feature, dtype=np.float32).reshape(2048, 36)).permute(1, 0)
    box = m.fetchone(colletion_id=true_feature_id, object_id=0) 
    boxes = torch.from_numpy(np.frombuffer(box, dtype=np.float32).reshape(4, 36)).permute(1, 0) 
    return features, boxes

if __name__ == "__main__":
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    gqa = GQA()
    gqa.model.cuda()
    if args.load is not None:
        gqa.load(args.load)
    gqa.model.eval()

    with torch.no_grad():
    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    #Change
        
    # vqa.to(device)

        batch_siz = 1024
        data1 = json.load(open('/playpen2/home/vaidehi/LXMERT-VQACP/data/gqa/train.json'))
        data2 = json.load(open('/playpen2/home/vaidehi/LXMERT-VQACP/data/gqa/valid.json'))
        data = data1 + data2
        print("Training data has {} samples".format(len(data1)+len(data2)))
        embs_v = torch.zeros(len(data),1,768)
        img = [data[i]['img_id'] for i in range(len(data))]
        print(img[:5])
        m = MIO(image_dataroot)
        ids = {}
        for i in range(m.size):
            id_= struct.unpack("<I", m.get_collection_metadata(i))[0]
            # print(id_)
            ids[id_] = i
        print(list(ids.keys())[:5])
        # img_batch = torch.stack(get_img_features_from_id(id) for id in img[i:i+batch_size])
        # vqa.model.lxrt_encoder(ques_batch, (torch.randn(len(ques_batch),36,2048), torch.randn(len(ques_batch),36,4)))

        for i in range(0,len(data),batch_siz):
              feat_list = [get_img_features_from_id(id) for id in img[i:i+batch_siz]]
              ques_batch = ["xyz"]*len(feat_list)
              img_batch_feat, img_batch_box = torch.stack([feat_list[j][0] for j in range(len(feat_list))]), torch.stack([feat_list[j][1] for j in range(len(feat_list))])
              img_batch_feat, img_batch_box = img_batch_feat.cuda(), img_batch_box.cuda()
              torch.manual_seed(1)
            #   print(len(vqa.model.lxrt_encoder(ques_batch, (img_batch_feat, img_batch_box)))
              embs_v[i:i+1024,0,:] = torch.mean(gqa.model.lxrt_encoder(ques_batch, (img_batch_feat, img_batch_box)),1).cpu()
            #   print(embs_v.shape)
            #   print(lang_feats)
            #   print(lang_feats.shape)
            #   exit()
            #   print(torch.mean(lang_feats,1).cpu().shape)
        torch.save(embs_v, '/playpen2/home/vaidehi/Deconfounded_disbiasing/vision_outputs_gqa_lxrt_orig.pt')
'''
'''
#For getting language features
if __name__ == "__main__":
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    gqa = GQA()
    gqa.model.cuda()
    if args.load is not None:
        gqa.load(args.load)
    gqa.model.eval()
    print(torch.cuda.current_device())
    # print(torch.cuda.device_count())
    # print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])

    with torch.no_grad():
    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    #Change
        
    # vqa.to(device)
    


        data1 = json.load(open('/playpen2/home/vaidehi/LXMERT-VQACP/data/gqa/train.json'))
        data2 = json.load(open('/playpen2/home/vaidehi/LXMERT-VQACP/data/gqa/valid.json'))
        print("Training data has {} samples".format(len(data1)+len(data2)))
        embs = torch.zeros(len(data1)+len(data2),1,768)
        questions = [data1[i]['sent'] for i in range(len(data1))]+[data2[i]['sent'] for i in range(len(data2))]

        for i in range(0,len(questions),1024):
              ques_batch =  questions[i:i+1024]
              torch.manual_seed(1)
              lang_feats = gqa.model.lxrt_encoder(ques_batch, (torch.randn(len(ques_batch),36,2048).cuda(), torch.randn(len(ques_batch),36,4).cuda()))
            #   print(lang_feats)
            #   print(lang_feats.shape)
            #   exit()
            #   print(torch.mean(lang_feats,1).cpu().shape)
              embs[i:i+1024,0,:]=torch.mean(lang_feats,1).cpu()
        torch.save(embs, '/playpen2/home/vaidehi/Deconfounded_disbiasing/language_outputs_gqa_lxrt_orig.pt')
'''


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cudnn.benchmark = False

    # Build Class
    gqa = GQA()

    # Load Model
    if args.load is not None:
        gqa.load(args.load)

    if args.wandb:
        wandb.init(project="vqacp", entity="adyasha", config={"seed": args.seed})

    # for param in gqa.model.lxrt_encoder.model.bert.encoder.AE_cross_lang.parameters():
    #     param.requires_grad = False

    # Test or Train
    if args.test is not None:
        print("testing")
        print(args.test)
        args.fast = args.tiny = False       # Always loading all data in test
        if 'submit' in args.test:
            gqa.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'submit_predict.json')
            )
        if 'ood_testdev_all' in args.test:
            result = gqa.evaluate(
                get_tuple('ood_testdev_all', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'testdev_all_predict.json')
            )
        if 'testdev' in args.test:
            print(os.path.join(args.output, 'testdev_predict.json'))
            result = gqa.evaluate(
                get_tuple('testdev', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'testdev_predict.json')
            )
            print(result)
    else:
        # print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
        print('Splits in Train data:', gqa.train_tuple.dataset.splits)
        if gqa.valid_tuple is not None:
            print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (gqa.oracle_score(gqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        gqa.train(gqa.train_tuple, gqa.valid_tuple)


