CUDA_VISIBLE_DEVICES=3 python tools/infer_simple_minimalist_saver.py \
        --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
        --output-dir /home/drussel1/data/ADL/detections/5_27_19 \
        --image-ext jpg \
        --wts https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
        /home/drussel1/data/ADL/ADL_frames/P_02 --always-out --output-ext png
