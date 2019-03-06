CUDA_VISIBLE_DEVICES=3 python tools/infer_simple_mysaver.py \
        --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
        --output-dir /home/drussel1/data/ADL/new_mask_outputs/dataset_per_frame \
        --image-ext mp4 \
        --wts https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl 
        /home/drussel1/data/EIMP/videos --is-video --always-out --MOT-format --visualize --output-ext png
