#note the fact that the extension has to match the input files. Might remove sometime soon.
CUDA_VISIBLE_DEVICES=3, python tools/infer_simple_mysaver.py \
        --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
        --output-dir /home/drussel1/data/EIMP/new-EIMP-mask-RCNN-detections \
        --image-ext mp4 \
        --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
        /home/drussel1/data/EIMP/videos --is-video --always-out --output-ext jpg --MOT-format
