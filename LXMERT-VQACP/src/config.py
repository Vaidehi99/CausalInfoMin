# ----------------------running settings-------------------------- #
cp_data     = True      # using vqa-cp or not
version     = 'v2'      # 'v1' or 'v2'
loss_type   = 'bce'     # 'bce' or 'ce'
in_memory   = False     # load all the image feature in memory
use_miu     = False     # using loss re-scale or not

# ----------------------before-process data paths---------------- #
main_path       = 'data/' #'data/gqa'
qa_path         = main_path + 'vqa-cp/' if cp_data else main_path
qa_path        += version # questions and answers
cache_root      = qa_path + '/cache/'

# ----------------------image related paths------------------- #
ids_path        = 'data/img_ids/' # mscoco image ids
image_path      = main_path + 'mscoco/' # image paths
rcnn_path       = 'data/mscoco_imgfeat/'#'data/mscoco_imgfeat/'

# ----------------------running settings------------------------- #
image_dataset       = 'mscoco'
task                = 'OpenEnded' if not cp_data else 'vqacp'
test_split          = 'test'    # 'test-dev2015' or 'test2015'
min_occurence       = 9             # answer frequency less than min will be omitted
num_fixed_boxes     = 36            # max number of object proposals per image
output_features     = 2048          # number of features in each object proposal
