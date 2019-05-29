import sys
import os
import glob
import pdb

pdb.set_trace()
INPUT_DIR = "/home/drussel1/data/ADL/ADL_frames" # no trailing slash
OUTPUT_DIR = "/home/drussel1/data/ADL/detections/5_29_19" # no trailing slash

files = sorted(glob.glob("{}/*".format(INPUT_DIR)))

for file_ in files:
    vid_name = file_.split("/")[-1]
    output_file = "{}/{}.h5".format(OUTPUT_DIR,vid_name)
    run_string = "CUDA_VISIBLE_DEVICES=3 python tools/infer_simple_minimalist_saver.py --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml --output-file {} --image-ext jpg --wts https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl {} --always-out --output-ext png".format(output_file, file_)
    os.system(run_string)
