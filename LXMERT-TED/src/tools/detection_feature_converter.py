import os
import sys
import csv
import h5py
import json
import base64
import argparse
import numpy as np
from tqdm import tqdm
sys.path.append(os.getcwd())
csv.field_size_limit(sys.maxsize)

import src.utils as utils
import src.config as config


FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    # we fold the train and val together to support both vqa and cp
    if not args.test:
        name_set = 'trainval'
        split_set = ['train', 'val']
        tsv_feat = [config.rcnn_path + 'train2014_obj36.tsv',
                    config.rcnn_path + 'val2014_obj36.tsv']
    else:
        name_set = 'test'
        split_set = ['test']
        tsv_feat = [config.rcnn_path + 'test2015_obj36.tsv']
    print('dumping image features for {} set\n'.format(name_set))

    # load all image ids
    img_ids = []
    for split in split_set:
        split_ids_path = os.path.join(config.ids_path, split + '_ids.json')
        if os.path.exists(split_ids_path):
            img_ids += json.load(open(split_ids_path, 'r'))
        else:
            split_year = '2014' if not args.test else '2015'
            split_image_path = os.path.join(
                config.image_path, split + split_year)
            img_ids_dump = utils.load_imageid(split_image_path)
            json.dump(list(img_ids_dump), open(split_ids_path, 'w'))
            img_ids += json.load(open(split_ids_path, 'r'))

    # create h5 files
    output = h5py.File(os.path.join(
            config.rcnn_path, name_set + '36.h5'), 'w')
    features = output.create_dataset('features', (len(img_ids),
        config.num_fixed_boxes, config.output_features), 'f')
    boxes = output.create_dataset('boxes', (len(img_ids),
        config.num_fixed_boxes, 4), 'f')
    counter, indices = 0, {}
    print("reading tsv...")

    for path in tsv_feat:
        with open(path, 'r') as tsv_in_file:
            reader = csv.DictReader(
                tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
            for item in tqdm(reader, initial=counter, total=len(img_ids)):
                image_id = int(item['img_id'].split('_')[-1])
                if image_id in img_ids:
                    num_boxes = int(item['num_boxes'])
                    decode_config = [
                        ('objects_id', (num_boxes,), np.int64),
                        ('objects_conf', (num_boxes,), np.float32),
                        ('attrs_id', (num_boxes,), np.int64),
                        ('attrs_conf', (num_boxes,), np.float32),
                        ('boxes', (num_boxes, 4), np.float32),
                        ('features', (num_boxes, -1), np.float32),
                    ]
                    for key, shape, dtype in decode_config:
                        item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                        item[key] = item[key].reshape(shape)
                        item[key].setflags(write=False)

                    # Normalize the boxes (to 0 ~ 1)
                    item['boxes'] = item['boxes'].copy()
                    item['boxes'][:, (0, 2)] /= float(item['img_w'])
                    item['boxes'][:, (1, 3)] /= float(item['img_h'])
                    np.testing.assert_array_less(item['boxes'], 1 + 1e-5)
                    np.testing.assert_array_less(-item['boxes'], 0 + 1e-5)

                    img_ids.remove(image_id)
                    indices[image_id] = counter
                    boxes[counter, :, :] = item['boxes']
                    features[counter, :, :] = item['features']
                    counter += 1

    if len(img_ids) != 0:
        print("Warning: {}_image_ids is not empty".format(name_set))

    print("done!")
    json.dump(indices, open(os.path.join(
        config.ids_path, name_set + '36_imgid2idx.json'), 'w'))
    output.close()


if __name__ == '__main__':
    main()
