from __future__ import print_function

import argparse
import os
import sys
import time
import tensorflow as tf
import numpy as np
import shutil
from scipy import misc

from model import PSPNet101, PSPNet50
from tools import *

cityscapes_param = {'crop_size': [720, 720],
                    'num_classes': 19,
                    'model': PSPNet101}

SAVE_DIR = './output/market/'
SNAPSHOT_DIR = './model/'
IMG_DIR = 'F:/jinhao/PSPNet/input/market/train/'
#IMG_DIR = 'F:/jinhao/PSPNet/input/'
TRANSFERED_DIR = 'F:/jinhao/PSPNet/input/masked/'
def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--checkpoints", type=str, default=SNAPSHOT_DIR,
                        help="Path to restore weights.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))
def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                L.append(file)
    return L
def main():
    args = get_arguments()
    # load parameters
    param = cityscapes_param
    crop_size = param['crop_size']
    num_classes = param['num_classes']
    PSPNet = param['model']

    # preprocess images
    #get all img path
    List = file_name(IMG_DIR)
    img_path = IMG_DIR+List[0]
    img, filename = load_img(img_path)
    img_shape = tf.shape(img)
    h, w = (tf.maximum(crop_size[0], img_shape[0]), tf.maximum(crop_size[1], img_shape[1]))
    img = preprocess(img, h, w)

    # Create network.
    net = PSPNet({'data': img}, is_training=False, num_classes=num_classes)
    with tf.variable_scope('', reuse=True):
        flipped_img = tf.image.flip_left_right(tf.squeeze(img))
        flipped_img = tf.expand_dims(flipped_img, dim=0)
        net2 = PSPNet({'data': flipped_img}, is_training=False, num_classes=num_classes)
    raw_output = net.layers['conv6']

    # Do flipped eval or not
    if args.flipped_eval:
        flipped_output = tf.image.flip_left_right(tf.squeeze(net2.layers['conv6']))
        flipped_output = tf.expand_dims(flipped_output, dim=0)
        raw_output = tf.add_n([raw_output, flipped_output])

    # Predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1])
    raw_output_up = tf.argmax(raw_output_up, axis=3)
    pred = decode_labels(raw_output_up, img_shape, num_classes)

    # Init tf Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    restore_var = tf.global_variables()

    ckpt = tf.train.get_checkpoint_state(args.checkpoints)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')



    preds = sess.run(pred)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    misc.imsave(args.save_dir + filename, preds[0])
    print('finished transfer image_'+img_path)
    print('moving transfered image to another folder...')
    shutil.move(img_path, TRANSFERED_DIR)
    print('done!')

if __name__ == '__main__':
    main()
