import os
import json
import h5py

import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

import src.utils as utils
import src.config as config


def _create_entry(img, question, answer, question_type=None):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'answer': answer}

    if question_type is not None:
        entry['question_type'] = question_type
    return entry


def _load_dataset(cache_path, name, img_id2val):
    """ Load entries. img_id2val: dict {img_id -> val} ,
        val can be used to retrieve image or features.
    """
    train, val, test = False, False, False
    if name == 'train':
        train = True
    elif name == 'val':
        val = True
    else:
        test = True
    question_path = utils.path_for(
                train=train, val=val, test=test, question=True)
    print("Loading data from %s" % question_path)
    questions = json.load(open(question_path, 'r'))

    qid2qtype = defaultdict(lambda: None)
    if train:
        anns_path = utils.path_for(train=train, val=val, test=test, question=False, answer=True)
        print("Loading data from %s" % anns_path)
        anns = json.load(open(anns_path, 'r'))
        for ann in anns:
            qid2qtype[ann["question_id"]] = ann["question_type"]

    if not config.cp_data:
        questions = questions['questions']
    questions = sorted(questions, key=lambda x: x['question_id'])
    if test: # will be ignored anyway
        answers = [
            {'image_id': 0, 'question_id': 0,
            'labels': [], 'scores': []}
            for _ in range(len(questions))]
    else:
        answer_path = os.path.join(cache_path, '{}_target.json'.format(name))
        answers = json.load(open(answer_path, 'r'))
        answers = sorted(answers, key=lambda x: x['question_id'])
        utils.assert_eq(len(questions), len(answers))

    entries = []
    for question, answer in zip(questions, answers):
        if not test:
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer, qid2qtype[question["question_id"]]))
    return entries


def _load_mask(cache_path, name):
    """ Load answer mask per question type and convert it
        to tensor: dict {qt -> mask in tensor type}.
    """
    mask_path = os.path.join(cache_path, '{}_mask.json'.format(name))
    qt_dict = json.load(open(mask_path, 'r'))

    for qt in qt_dict:
        ans_mask = utils.json_keys2int(qt_dict[qt])
        # print(ans_mask.keys())
        qt_dict[qt] = {
            'mask': torch.from_numpy(np.array(
                list(ans_mask.keys()))),
            'weight': torch.from_numpy(np.array(
                list(ans_mask.values()), dtype=np.float32))
        }
    return qt_dict


class VQADataset:
    def __init__(self, name: str):
        self.name = name
        assert name in ['train', 'val', 'test']

        # loading image ids
        self.image_split = 'test' if name == 'test' else 'trainval'
        self.img_id2idx = json.load(open(os.path.join(
            config.ids_path, '{}36_imgid2idx.json'.format(
                self.image_split)), 'r'), object_hook=utils.json_keys2int)

        # Answers
        self.ans2label = json.load(open(os.path.join(
            config.cache_root, 'trainval_ans2label.json'), 'r'))
        self.label2ans = json.load(open(os.path.join(
            config.cache_root, 'trainval_label2ans.json'), 'r'))
        assert len(self.ans2label) == len(self.label2ans)

        self.entries = _load_dataset(config.cache_root, name, self.img_id2idx)

    @property
    def num_answers(self):
        return len(self.ans2label)

    @property
    def v_dim(self):
        return config.output_features

    @property
    def s_dim(self):
        return config.num_fixed_boxes

    def __len__(self):
        return len(self.entries)


class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset):
        super().__init__()
        self.dataset = dataset

        # loading image features
        self.h5_path = os.path.join(config.rcnn_path,
                                    '{}36.h5'.format(dataset.image_split))
        if config.in_memory:
            print('loading image features from h5 file')
            with h5py.File(self.h5_path, 'r') as hf:
                self.features = np.array(hf.get('features'))
                self.spatials = np.array(hf.get('boxes'))

        self.answer_mask = _load_mask(config.cache_root, dataset.name)

        self.tensorize()

    @property
    def num_answers(self):
        return self.dataset.num_answers

    @property
    def v_dim(self):
        return config.output_features

    @property
    def s_dim(self):
        return config.num_fixed_boxes

    def __len__(self):
        return self.dataset.__len__()

    def tensorize(self):
        if config.in_memory:
            self.features = torch.from_numpy(self.features)
            self.spatials = torch.from_numpy(self.spatials)

        for entry in self.dataset.entries:
            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def load_image(self, image_id):
        """ Load one image feature. """
        if not hasattr(self, 'image_feat'):
            self.image_feat = h5py.File(self.h5_path, 'r')
        features = self.image_feat['features'][image_id]
        spatials = self.image_feat['boxes'][image_id]
        return torch.from_numpy(features), torch.from_numpy(spatials)

    def __getitem__(self, index):
        entry = self.dataset.entries[index]
        if config.in_memory:
            features = self.features[entry['image']]
            spatials = self.spatials[entry['image']]
        else:
            features, spatials = self.load_image(entry['image'])

        question_id = entry['question_id']
        question = entry['question']
        answer = entry['answer']
        q_type = answer['question_type']
        labels = answer['labels']
        scores = answer['scores']
        mask_labels = self.answer_mask[q_type]['mask']
        mask_scores = self.answer_mask[q_type]['weight']

        target = torch.zeros(self.num_answers)
        target_mask = torch.zeros(self.num_answers)
        target_score = torch.ones(self.num_answers)
        if labels is not None:
            target.scatter_(0, labels, scores)
            target_mask.scatter_(0, mask_labels, 1.0)
            target_score.scatter_(0, mask_labels, mask_scores)

        return question_id, features, spatials, question, target,\
                                        target_mask, target_score


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.

        results = []

        for i, (quesid, ans) in enumerate(sorted(quesid2ans.items())):
            entry = self.dataset.entries[i]
            utils.assert_eq(entry['question_id'], quesid)
            label = entry['answer']['labels']
            if label is not None:
                label = label.tolist()
                if type(ans) == str:
                    label = [self.dataset.label2ans[k] for k in label]
                if ans in label:
                    score += entry['answer']['scores'][label.index(ans)]

                if ans in label:
                    acc = entry['answer']['scores'][label.index(ans)].item()
                else:
                    acc = 0.0
                entry = {
                    'question_id': quesid,
                    'answer_true': label[np.array(entry['answer']['scores']).argmax()],
                    'answer_pred': ans,
                    'accuracy': acc,
                }
                results.append(entry)

        return score / len(quesid2ans), results

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)

    def dump_results(self, quesid2ans: list, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for data_dict in quesid2ans:
                result.append({
                    'question_id': data_dict["qid"],
                    'answer': data_dict["answer"],
                    'confidence': data_dict["conf"]
                })
            json.dump(result, f, indent=4)
        print("Saved predictions for %s samples" % len(quesid2ans))
