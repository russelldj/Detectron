#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import json
import h5py
import numpy as np

from caffe2.python import workspace
import pycocotools.mask as mask_util

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--is-video',
        dest='is_video',
        help='input is in video format',
        action='store_true'
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='pdf',
        type=str
    )
    parser.add_argument(
        '--visualize',
        dest='visualize',
        help='write the detection images',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--MOT-format',
        dest='mot_format',
        help='whether to output detections in the format specified by the MOT challenge',
        action='store_true'

    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)


    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        # MOD
        print(args.im_or_folder)
        im_list = sorted(glob.iglob(args.im_or_folder + '/*.' + args.image_ext))
        print(im_list)
    else:
        im_list = [args.im_or_folder]

    
    #I'm going to have a massive repetition of code because I can't think of a cleaner way to do it
    def infer(im, i, output_name='None', video='None', mot_output=[]):
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers)
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )
        boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)

        if segms is not None: 
            masks = mask_util.decode(segms)
        else:
            masks = np.asarray([[[]]]) # an empty array with shape[2] == 0
        all_contours = []
        for mask_idx in range(masks.shape[2]):
            print("shapes are {}".format(masks[...,mask_idx].shape))
            _, contours, _ = cv2.findContours(masks[...,mask_idx].copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.append(contours) # this code is more general and allows for multiple contours, but there aren't any

        if boxes is None:
            boxes = []
        else:
            boxes = boxes.tolist()
            print(classes)

        # create the mot formated row
        def mot_row(i, boxes, classes):
            """<frame>, <id=class>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x=-1>, <y=-1>, <z=-1>
            """
            assert len(boxes) == len(classes), "the boxes weren't the same length as the boxes"
            out_ = np.empty((0,10), float)
            for box_id, box_ in enumerate(boxes):
                class_ = classes[box_id]
                # check that the conversion is correct
                # and that conf is where I think it is
                out_ = np.append(out_, np.array([[i,classes[box_id], box_[0], box_[1], box_[2]-box_[0], box_[3]-box_[1], box_[4], -1.0, -1.0, -1.0]]), axis=0)
            return out_

        if args.mot_format:
            #TODO actually write out the class o fthe detection
            try:
                mot_output = np.append(mot_output, mot_row(i, boxes, classes), axis=0)
            except ValueError:
                import pdb; pdb.set_trace()
            json_output.append({
                'video': video,
                'frame': i,
                'boxes': boxes,
                'classes': classes,
                'contours': [[c.tolist() for c in some_contours] for some_contours in all_contours]
            })
            masks_output[i] = [[c.tolist() for c in some_contours] for some_contours in all_contours]# this will be a mapping from frame to contours
        else:
            json_output.append({
                'video': video,
                'frame': i,
                'boxes': boxes,
                'classes': classes,
                'contours': [[c.tolist() for c in some_contours] for some_contours in all_contours]
            })

        if i % 1000 == 0:
            save_files(args, im_name, output_basename, mot_output, json_output, masks_output)

        # MOD
        print('output name {}'.format(output_name))
        #HACK 
        VISUALIZATION_FREQUENCY = 100
        if args.visualize and i % VISUALIZATION_FREQUENCY == 0:
            start_time = time.time()
            vis_utils.vis_one_image(
                im[:, :, ::-1],  # BGR -> RGB for visualization
                output_name,
                "{}/images".format(args.output_dir),
                cls_boxes,
                cls_segms,
                cls_keyps,
                dataset=dummy_coco_dataset,
                box_alpha=0.3,
                show_class=True,
                thresh=0.4, # change the display confidence
                kp_thresh=2,
                ext=args.output_ext,
                out_when_no_box=args.out_when_no_box
            )
            print("only visualizing every {}th frame, it took {} seconds".format(VISUALIZATION_FREQUENCY, time.time() - start_time))
        if args.mot_format:
            return mot_output
        else:
            pass
            # likely something should go here
        # end of infer

    print('im_list is {}'.format(im_list))
    if args.is_video:
        for vid_ind, im_name in enumerate(im_list):
            # initialize the outputs
            if args.mot_format:
                # do some initialization, maybe the name of the file
                mot_output = np.empty((0,10), float)#[[-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,1.0]]
                masks_output = {}
                json_output = []
            else:
                json_output = []

            cap = cv2.VideoCapture(im_name)
            ret, im = cap.read() 
            i = 0
            while ret:
                out_name = os.path.join(
                    args.output_dir, '{}_{:09d}.{}'.format(os.path.basename(im_name), i, args.output_ext)
                )
                logger.info('Processing {} -> {}'.format(im_name, out_name))
                output_basename = '{}_{:09d}'.format(os.path.basename(im_name), i)
                mot_output = infer(im, i, output_basename, im_name, mot_output) # this writes the images so it is pretty hacky
                i += 1
                ret, im = cap.read() 

            #write the data out
            #TODO get pylinter

    else:
        for i, im_name in enumerate(im_list):
            out_name = os.path.join(
                args.output_dir, '{}'.format(os.path.basename(im_name) + '.' + args.output_ext)
            )
            logger.info('processing {} -> {}'.format(im_name, out_name))
            im = cv2.imread(im_name)
            infer(im, i, im_name)

        save_files(args, im_name, output_basename, mot_output, json_output, masks_output=None)


def save_files(args, im_name, output_basename, mot_output, json_output, masks_output):
    if args.mot_format:
        print(mot_output.shape)
        #raw_input('{}/{}.npy'.format(os.path.basename(im_name)))
        np.save('{}/{}.npy'.format(args.output_dir, os.path.basename(im_name)), mot_output)

        with h5py.File("{}/{}.h5".format(args.output_dir,os.path.basename(im_name)), 'w') as outfile:
             #outfile.create_dataset("data", data=json.dumps(json_output))
             for frame_data in json_output:
                 frame_id = frame_data["frame"]
                 if not str(frame_id) in outfile: # check if it's in the keys
                     outfile.create_dataset(str(frame_id), data=json.dumps(frame_data))
             json_output = []


        #with h5py.File("{}/{}_only_masks.h5".format(args.output_dir,os.path.basename(im_name)), 'w') as outfile:
        #    #TODO the json data should only be encoded once
        #    outfile.create_dataset("data", data=json.dumps(masks_output))# TODO this can be decoded as json.loads(h5py.File("filename")["data"].value)
        
    else:
        with open("{}/{}.json".format(args.output_dir,os.path.basename(im_name)), 'w') as outfile:
            json.dump(json_output, outfile, indent=4)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    #raw_input(args.mot_format)
    main(args)
