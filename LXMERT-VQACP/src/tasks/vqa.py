import sys
sys.path.append("/nas-ssd2/vaidehi/cache")
import os
import collections
from tqdm import tqdm
from pprint import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
#Change
# import numpy as np
import json
# import torch.nn.functional as F
# import sys
# import struct
# sys.path.append("/ssd-playpen/home/vaidehi/vaidehi/D-VQA/LXMERT")
# from mio import MioWriter, MIO
# sys.path.append("/ssd-playpen/home/vaidehi/vaidehi/LXMERT-VQACP/src")
#Change

from src.param import args
import src.config as config

from src.tasks.vqa_model import VQAModel_original 
from src.tasks.vqa_model import VQAModel_inference 
from src.lxrt.optimization import BertAdam
from src.tasks.losses import FocalLoss, Plain
from src.pretrain.qa_answer_table import load_lxmert_qa
from src.tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')



def get_data_tuple(splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:
    dset = VQADataset(splits)
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)
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
        self.model = VQAModel_original(self.train_tuple.dataset.num_answers)
        # self.model_inference = VQAModel_inference(self.train_tuple.dataset.num_answers)

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
            self.loss_fn = Plain()
        elif args.loss_fn == 'Focal':
            self.loss_fn = FocalLoss()
        else:
            raise RuntimeError('not implement for {}'.format(args.loss_fn))
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target, mask, score) in iter_wrapper(enumerate(loader)):
                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit, deconf_loss = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2

                dict_args = {}
                if config.use_miu:
                    dict_args['miu'] = score.cuda()
                    dict_args['mask'] = mask.cuda()
                # print(deconf_loss)
                if(epoch>=5):
                    loss = self.loss_fn(logit, target, **dict_args) + 0.2*deconf_loss
                else:
                    loss = self.loss_fn(logit, target, **dict_args)
                # print(loss)
                # print(loss - 0.8*deconf_loss)
                # exit()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, ans in zip(ques_id, label.cpu().numpy()):
                    quesid2ans[qid.item()] = ans

            print("\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans)[0] * 100.))

            if self.valid_tuple is not None:  # Do Validation
                # valid_score = self.evaluate(eval_tuple)
                valid_score, save_results = self.evaluate(eval_tuple)
                if valid_score > best_valid:

                    best_results = save_results

                    best_valid = valid_score
                    self.save(args.name, epoch, best_valid)

                print("Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) +
                      "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.))

        import json
        cp = 'CP' if config.cp_data else 'NC'
        miu = 'miu' if config.use_miu else 'base'
        with open('./result_{}_{}_{}.json'.format(
                cp, config.version, miu), 'w') as fd:
            json.dump(best_results, fd)

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}

        for i, datum_tuple in tqdm(enumerate(loader), total=len(loader)):
            ques_id, feats, boxes, sent, target = datum_tuple[:5]
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit, deconf_loss = self.model(feats, boxes, sent)
                # Change
                # print(len(sent))
                # print(torch.exp(logit).sum(1))
                # print(logit.max(1))
                # exit()
                # Change
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def predict_inference(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model_inference.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}

        for i, datum_tuple in tqdm(enumerate(loader), total=len(loader)):
            ques_id, feats, boxes, sent, target = datum_tuple[:5]
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit, deconf_loss = self.model_inference(feats, boxes, sent)
                # Change
                # print(len(sent))
                # print(torch.exp(logit).sum(1))
                # print(logit.max(1))
                # exit()
                # Change
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    def evaluate_inference(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict_inference(eval_tuple, dump)
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
        print("Load model from %s" % path)
        state_dict = torch.load("%s" % path)
        self.model.load_state_dict(state_dict['model_state'])
        self.optim.load_state_dict(state_dict['optim_state'])
        self.loss_fn.load_state_dict(state_dict['loss_state'])
        # self.model_inference.load_state_dict(state_dict['model_state'])

    def load_some(self, path):
        print("Load model partially from %s" % path)
        state_dict = torch.load("%s" % path)
        #self.optim.load_state_dict(state_dict['optim_state'])
        #self.loss_fn.load_state_dict(state_dict['loss_state'])
        own_state = self.model.state_dict()
        for name, param in state_dict['model_state'].items():
            if name not in own_state:
                 print(name)
                 continue
            #if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
            else:
                param = param.data
            own_state[name].copy_(param)

'''
#"for getting substitute confounders lang"

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        # self.encoder = torch.nn.Sequential(
        #     torch.nn.Linear(768, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, 64),
        # )
        self.param = nn.Parameter(torch.randn(64, 768))
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        # self.decoder = torch.nn.Sequential(
        #     torch.nn.Linear(64, 128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, 768)
        # )
 
    def forward(self, x):
        # encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        # return decoded
        encoded_feats = F.linear(x, self.param)
        reconstructed_output = F.linear(encoded_feats, self.param.t())
        return encoded_feats, reconstructed_output

if __name__ == "__main__":
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    vqa = VQA()
    vqa.model.cuda()
    vqa.model.eval()
    print(torch.cuda.current_device())
    # print(torch.cuda.device_count())
    # print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    AE_lang_path = "/ssd-playpen/home/vaidehi/vaidehi/Deconfounded_disbiasing/language_lxmert_only_AE.pt"
    AE_lang = AE()
    AE_lang.load_state_dict(torch.load(AE_lang_path,map_location='cuda:'+str(0)))
    AE_lang.eval()
    AE_lang=AE_lang.cuda()
    with torch.no_grad():
    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    #Change
        
    # vqa.to(device)
    


        data = json.load(open('/ssd-playpen/home/vaidehi/vaidehi/D-VQA/data/vqacp2/vqacp_v2_train_questions.json'))
        print("Training data has {} samples".format(len(data)))
        embs = torch.zeros(len(data),20,64)
        questions = [data[i]['question'] for i in range(len(data))]

        for i in range(0,len(questions),1024):
              ques_batch =  questions[i:i+1024]
              torch.manual_seed(1)
              lang_feats = vqa.model.lxrt_encoder(ques_batch, (torch.randn(len(ques_batch),36,2048).cuda(), torch.randn(len(ques_batch),36,4).cuda()))
            #   print(lang_feats.shape)
            #   exit()
            #   lang_feats=lang_feats.cuda()
            #   print(lang_feats.shape)
              subs_lang = AE_lang(lang_feats.cuda())[0]
            #   print(subs_lang.shape)
            #   print(lang_feats)
            #   print(lang_feats.shape)
            #   exit()
            #   print(torch.mean(lang_feats,1).cpu().shape)
              embs[i:i+1024,:,:]=subs_lang.cpu()
        torch.save(embs, '/ssd-playpen/home/vaidehi/vaidehi/Deconfounded_disbiasing/language_sub_conf_AE_lang_lxmert_only.pt')

'''
if __name__ == "__main__":
    print(args)
    print_keys = ['cp_data', 'version', 'use_miu']
    print_dict = {key: getattr(config, key) for key in print_keys}
    pprint(print_dict, width=150)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    # Build Class
    vqa = VQA()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # for param in vqa.model.lxrt_encoder.parameters():
    #     param.requires_grad = False

    # for param in vqa.model.logit_fc.parameters():
    #     param.requires_grad = False
    
    # for param in vqa.model.lxrt_encoder.model.bert.encoder.DA_lang.parameters():
    #     param.requires_grad = True

     
    # for param in vqa.model.lxrt_encoder.model.bert.encoder.layer.parameters():
    #     param.requires_grad = False

    # for param in vqa.model.lxrt_encoder.model.bert.encoder.x_layers.parameters():
    #     param.requires_grad = False

    # for param in vqa.model.lxrt_encoder.model.bert.embeddings.parameters():
    #     param.requires_grad = False

    # for param in vqa.model.lxrt_encoder.model.bert.encoder.visn_fc.parameters():
    #     param.requires_grad = False

    for param in vqa.model.lxrt_encoder.model.bert.encoder.AE_cross_lang.parameters():
        param.requires_grad = False

    # for param in vqa.model.lxrt_encoder.model.bert.encoder.AE_vision.parameters():
    #     param.requires_grad = False

    

    
    

        

    # print(vqa.model)
    print("Parameters requiring gradient")
    for name, param in vqa.model.named_parameters():
        if param.requires_grad:
            print(name)
    # exit()
    
    # Test or Train
    print("args.test")
    print(args.test)
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
        elif 'inf' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate_inference(
                get_data_tuple('val', bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'val_predict.json')
            )
        else:
            assert False, "No such test option for %s" % args.test
    else:
        vqa.train(vqa.train_tuple, vqa.valid_tuple)


'''
#For getting language-image features
image_dataroot="/ssd-playpen/home/vaidehi/vaidehi/D-VQA/data/trainval_features_with_boxes" 

def get_img_features_from_id(img_id):
    true_feature_id = ids[img_id]
    feature = m.fetchone(colletion_id=true_feature_id, object_id=1)
    features = torch.from_numpy(np.frombuffer(feature, dtype=np.float32).reshape(2048, 36)).permute(1, 0)
    box = m.fetchone(colletion_id=true_feature_id, object_id=0) 
    boxes = torch.from_numpy(np.frombuffer(box, dtype=np.float32).reshape(4, 36)).permute(1, 0) 
    return features, boxes
   

if __name__ == "__main__":
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    vqa = VQA()
    vqa.model.eval()
    # print(torch.cuda.current_device())
    # print(torch.cuda.device_count())
    # print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    with torch.no_grad():
    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    #Change
        
    # vqa.to(device)
    

        batch_siz = 1024
        data = json.load(open('/ssd-playpen/home/vaidehi/vaidehi/D-VQA/data/vqacp2/vqacp_v2_train_questions.json'))
        print("Training data has {} samples".format(len(data)))
        x_embs_l = torch.zeros(len(data),1,768)
        x_embs_v = torch.zeros(len(data),1,768)
        img = [data[i]['image_id'] for i in range(len(data))]
        questions = [data[i]['question'] for i in range(len(data))]
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
            #   print(vqa.model.lxrt_encoder(ques_batch, (img_batch_feat, img_batch_box)).shape)
              (x_lang_feats, x_img_feats), _ = vqa.model.lxrt_encoder(ques_batch, (img_batch_feat, img_batch_box))
            #   print(lang_feats)
            #   print(lang_feats.shape)
            #   exit()
            #   print(torch.mean(lang_feats,1).cpu().shape)
              x_embs_l[i:i+1024,0,:] = torch.mean(x_lang_feats,1).cpu()
              x_embs_v[i:i+1024,0,:] = torch.mean(x_img_feats,1).cpu()
        torch.save(x_embs_l, '/ssd-playpen/home/vaidehi/vaidehi/Deconfounded_disbiasing/cross_language_outputs_lxmert_only_2.pt')
        torch.save(x_embs_v, '/ssd-playpen/home/vaidehi/vaidehi/Deconfounded_disbiasing/cross_vision_outputs_lxmert_only_2.pt')



#For getting language features
if __name__ == "__main__":
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    vqa = VQA()
    vqa.model.cuda()
    vqa.model.eval()
    print(torch.cuda.current_device())
    # print(torch.cuda.device_count())
    # print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    with torch.no_grad():
    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    #Change
        
    # vqa.to(device)
    


        data = json.load(open('/ssd-playpen/home/vaidehi/vaidehi/D-VQA/data/vqacp2/vqacp_v2_train_questions.json'))
        print("Training data has {} samples".format(len(data)))
        embs = torch.zeros(len(data),1,768)
        questions = [data[i]['question'] for i in range(len(data))]

        for i in range(0,len(questions),1024):
              ques_batch =  questions[i:i+1024]
              torch.manual_seed(1)
              lang_feats = vqa.model.lxrt_encoder(ques_batch, (torch.randn(len(ques_batch),36,2048).cuda(), torch.randn(len(ques_batch),36,4).cuda()))
            #   print(lang_feats)
            #   print(lang_feats.shape)
            #   exit()
            #   print(torch.mean(lang_feats,1).cpu().shape)
              embs[i:i+1024,0,:]=torch.mean(lang_feats,1).cpu()
        torch.save(embs, '/ssd-playpen/home/vaidehi/vaidehi/Deconfounded_disbiasing/language_outputs_lxmert_only_2.pt')
'''
'''
#For getting image features
image_dataroot="/ssd-playpen/home/vaidehi/vaidehi/D-VQA/data/trainval_features_with_boxes" 

def get_img_features_from_id(img_id):
    true_feature_id = ids[img_id]
    feature = m.fetchone(colletion_id=true_feature_id, object_id=1)
    features = torch.from_numpy(np.frombuffer(feature, dtype=np.float32).reshape(2048, 36)).permute(1, 0)
    box = m.fetchone(colletion_id=true_feature_id, object_id=0) 
    boxes = torch.from_numpy(np.frombuffer(box, dtype=np.float32).reshape(4, 36)).permute(1, 0) 
    return features, boxes

if __name__ == "__main__":
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    vqa = VQA()
    vqa.model.eval()

    with torch.no_grad():
    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    #Change
        
    # vqa.to(device)

        batch_siz = 1024
        data = json.load(open('/ssd-playpen/home/vaidehi/vaidehi/D-VQA/data/vqacp2/vqacp_v2_train_questions.json'))
        print("Training data has {} samples".format(len(data)))
        embs_v = torch.zeros(len(data),1,768)
        img = [data[i]['image_id'] for i in range(len(data))]
        m = MIO(image_dataroot)
        ids = {}
        for i in range(m.size):
            id_= struct.unpack("<I", m.get_collection_metadata(i))[0]
            ids[id_] = i

        # img_batch = torch.stack(get_img_features_from_id(id) for id in img[i:i+batch_size])
        # vqa.model.lxrt_encoder(ques_batch, (torch.randn(len(ques_batch),36,2048), torch.randn(len(ques_batch),36,4)))

        for i in range(0,len(data),batch_siz):
              feat_list = [get_img_features_from_id(id) for id in img[i:i+batch_siz]]
              ques_batch = ["xyz"]*len(feat_list)
              img_batch_feat, img_batch_box = torch.stack([feat_list[j][0] for j in range(len(feat_list))]), torch.stack([feat_list[j][1] for j in range(len(feat_list))])
              img_batch_feat, img_batch_box = img_batch_feat.cuda(), img_batch_box.cuda()
              torch.manual_seed(1)
            #   print(len(vqa.model.lxrt_encoder(ques_batch, (img_batch_feat, img_batch_box)))
              embs_v[i:i+1024,0,:] = torch.mean(vqa.model.lxrt_encoder(ques_batch, (img_batch_feat, img_batch_box)),1).cpu()
            #   print(embs_v.shape)
            #   print(lang_feats)
            #   print(lang_feats.shape)
            #   exit()
            #   print(torch.mean(lang_feats,1).cpu().shape)
        torch.save(embs_v, '/ssd-playpen/home/vaidehi//vaidehi/Deconfounded_disbiasing/vision_outputs_lxmert_only_2.pt')

'''
