#note the fact that the extension has to match the input files. Might remove sometime soon.
python -m pdb tools/infer_simple_mysaver.py \
        --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
        --output-dir /scratch/drussel1/EPIC_output \
        --image-ext MP4 \
        --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
        /scratch/drussel1/EPIC_subset --is-video --always-out --output-ext jpg --MOT-format
