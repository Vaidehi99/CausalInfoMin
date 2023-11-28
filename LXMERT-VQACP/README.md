# LXMERT for VQA-CP and VQA.

This repo made a few modifications to support both VQA-CP and VQA datasets. Please find more details at the original LXMERT [code](https://github.com/airsplay/lxmert).

We mainly use this repo to implement our paper - Loss Re-scaling VQA: Revisiting the Language Prior Problem from a Class-imbalance View.  

# Pre-trained models

The pre-trained model (870 MB) is available at http://nlp.cs.unc.edu/data/model_LXRT.pth, and can be downloaded with:

```
mkdir -p snap/pretrained 
wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P snap/pretrained
```

# Fine-tuning on VQA-CP or VQA

1. Please make sure the LXMERT pre-trained model is either downloaded or pre-trained.

2. Note that we DO NOT use the re-distributed json file provided by LXMERT authors. We use the official splits in this repo. Make sure that these data are in the right position according to the ```src/config.py```!
  
3. Download faster-rcnn features for MS COCO train2014 (17 GB) and val2014 (8 GB) images (VQA 2.0 is collected on MS COCO dataset). 
    ```
    mkdir -p data/mscoco_imgfeat
    wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/mscoco_imgfeat
    unzip data/mscoco_imgfeat/train2014_obj36.zip -d data/mscoco_imgfeat && rm data/mscoco_imgfeat/train2014_obj36.zip
    wget https://nlp.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/mscoco_imgfeat
    unzip data/mscoco_imgfeat/val2014_obj36.zip -d data && rm data/mscoco_imgfeat/val2014_obj36.zip
    ```
4. We convert the image features from ```tsv``` to ```h5``` first:
    ```
    python src/tools/detection_feature_converter.py
    ```
    We fold the train and val image features together for supporting both VQA-CP and VQA.

5. Process answers and question types:
    ```
    python src/tools/compute_softscore.py 
    ```
6. Fine-tuning on VQA-CP or VQA (set this on the ```src/config.py```):
    ```
    PYTHONPATH=$PYTHONPATH:./src \
    python -u src/tasks/vqa.py \
    --train train --valid val  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA snap/pretrained/model \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
    --tqdm
    --name vqa-cp-test
    ```
7. Evaluating on the validation set (according to the official implementation):
    ```
    PYTHONPATH=$PYTHONPATH:./src \
    python -u src/tasks/vqa.py \
    --train train --test val  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA snap/pretrained/model \
    --batchSize 32 --load output/vqa-cp-test.pth \
    --tqdm
    ```
    ```
    python acc_per_type.py output/val_predict.json
    ```
## Performance on VQA-CP test

<center>

Loss Function   | Model         | Y/N   | Num.  | Others    | All
-------         | ------------- | ----- | ----- | --------- | -----
BCE             | LXMERT        |46.70  | 27.14 | 61.20     | 51.78
BCE             | LXMERT+Ours   |79.77  | 59.06 | 61.41     | 66.40
CE              | LXMERT        |-      | -     | -         | 58.07
CE              | LXMERT+Ours   |-      | -     | -         | 69.37
                
</center>

## Citation
If you found this repo useful, please cite the following paper:  
```
@article{rescale-vqa,
  title={Loss Re-scaling VQA: Revisiting the Language Prior Problem from a Class-imbalance View},
  author={Guo, Yangyang and Nie, Liqiang and Cheng, Zhiyong and Tian, Qi and Zhang, Min},
  journal={IEEE TIP},
  year={2021}
}
```  
