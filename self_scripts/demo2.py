# pylint: skip-file
import argparse
import cPickle
import os
import os.path as osp
import re
import sys
# sys.path.insert(0, '/opt/github/ademxapp')
# sys.path.insert(0, '/opt/github/ademxapp/util')
# sys.path.insert(0, '/opt/github/ademxapp/issegm')
# sys.path.insert(0, '/opt/github/ademxapp/data/cityscapesScripts/cityscapesscripts/helpers')
import time
from functools import partial
from PIL import Image
from multiprocessing import Pool

import numpy as np

import mxnet as mx

from util import mxutil
from util import transformer as ts
from util import util

from util.symbol.resnet_v2 import fcrna_model_a1


def make_divisible(v, divider):
    return int(np.ceil(float(v) / divider) * divider)


def parse_args():
    parser = argparse.ArgumentParser(description='Tune FCRNs from ResNets.')
    parser.add_argument('--gpus', default='0',
                        help='The devices to use, e.g. 0,1,2,3')
    parser.add_argument('--data-root', dest='data_root',
                        help='The root data dir.',
                        default=None, type=str)
    parser.add_argument('--file_list',
                        default=None, type=str)
    parser.add_argument('--output', default=None,
                        help='The output dir.')
    parser.add_argument('--model', default=None,
                        help='The unique label of this model.')
    parser.add_argument('--weights', default=None,
                        help='The path of a pretrained model.')
    parser.add_argument('--test-flipping', dest='test_flipping',
                        help='If average predictions of original and flipped images.',
                        default=False, action='store_true')
    parser.add_argument('--log-file', dest='log_file',
                        default=None, type=str)
    parser.add_argument('--start', dest='log_file',
                        default=0, type=int)
    parser.add_argument('--end', dest='log_file',
                        default=0, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    if args.output:
        if not os.path.exists(args.output):
            os.makedirs(args.output)

    if args.file_list:
        if not os.path.exists(args.file_list):
            parser.print_help()
            sys.exit(1)

    if args.weights is not None:
        #
        if args.model is None:
            assert '_ep-' in args.weights
            parts = osp.basename(args.weights).split('_ep-')
            args.model = '_'.join(parts[:-1])

    if args.model is None:
        raise NotImplementedError('Missing argument: args.model')

    args.log_file = '{}/mxnet-cityscapes.log'.format(args.output)

    model_specs = {
        # model
        'net_type': 'rna',
        'net_name': 'a1',
        'classes': 19,
        'feat_stride': 8,
        # data
        'dataset': "cityscapes",
    }
    if args.data_root is None:
        args.data_root = osp.join('data', model_specs['dataset'])

    return args, model_specs


def get_dataset_specs(args, model_specs):
    dataset = model_specs['dataset']
    meta = {}
    meta_path = osp.join('issegm/data', dataset, 'meta.pkl')
    if osp.isfile(meta_path):
        with open(meta_path) as f:
            meta = cPickle.load(f)

    label_2_id = None
    id_2_label = None
    ident_size = False
    cache_images = False
    mx_workspace = 1650
    if dataset == 'cityscapes':
        sys.path.insert(0, 'data/cityscapesScripts/cityscapesscripts/helpers')
        from labels import id2label, trainId2label
        #
        label_2_id = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            label_2_id[l] = id2label[l].trainId
        id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
        valid_labels = sorted(set(id_2_label.ravel()))
        #
        cmap = np.zeros((256, 3), dtype=np.uint8)
        for i in id2label.keys():
            cmap[i] = id2label[i].color
        #
        ident_size = True
        #
        max_shape = np.array((1024, 2048))
        mx_workspace = 8000

    meta['label_2_id'] = label_2_id
    meta['id_2_label'] = id_2_label
    meta['valid_labels'] = valid_labels
    meta['cmap'] = cmap
    meta['ident_size'] = ident_size
    meta['max_shape'] = meta.get('max_shape', max_shape)
    meta['cache_images'] = cache_images
    meta['mx_workspace'] = mx_workspace
    return meta


def _get_metric():
    def _eval_func(label, pred):
        gt_label = label.ravel()
        valid_flag = gt_label != 255
        gt_label = gt_label[valid_flag]
        pred_label = pred.argmax(1).ravel()[valid_flag]

        sum_metric = (gt_label == pred_label).sum()
        num_inst = valid_flag.sum()
        return (sum_metric, num_inst + (num_inst == 0))

    return mx.metric.CustomMetric(_eval_func, 'fcn_valid')


def _get_scalemeanstd():
    if model_specs['net_type'] == 'rn':
        return -1, np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3)), None
    if model_specs['net_type'] in ('rna',):
        return (1.0 / 255,
                np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),
                np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)))
    return None, None, None


def _get_transformer_image():
    scale, mean_, std_ = _get_scalemeanstd()
    transformers = []
    if scale > 0:
        transformers.append(ts.ColorScale(np.single(scale)))
    transformers.append(ts.ColorNormalize(mean_, std_))
    return transformers


def _interp_preds_as_impl(num_classes, im_size, pred_stride, imh, imw, pred):
    imh0, imw0 = im_size
    pred = pred.astype(np.single, copy=False)
    input_h, input_w = pred.shape[0] * pred_stride, pred.shape[1] * pred_stride
    assert pred_stride >= 1.
    this_interp_pred = np.array(Image.fromarray(pred).resize((input_w, input_h), Image.CUBIC))
    if imh0 == imh:
        interp_pred = this_interp_pred[:imh, :imw]
    else:
        interp_method = util.get_interp_method(imh, imw, imh0, imw0)
        interp_pred = np.array(Image.fromarray(this_interp_pred[:imh, :imw]).resize((imw0, imh0), interp_method))
    return interp_pred


def interp_preds_as(im_size, net_preds, pred_stride, imh, imw, threads=4):
    num_classes = net_preds.shape[0]
    worker = partial(_interp_preds_as_impl, num_classes, im_size, pred_stride, imh, imw)
    if threads == 1:
        ret = [worker(_) for _ in net_preds]
    else:
        pool = Pool(threads)
        ret = pool.map(worker, net_preds)
        pool.close()
    return np.array(ret)


# @profile
def do(args, model_specs, logger):
    meta = get_dataset_specs(args, model_specs)
    id_2_label = meta['id_2_label']
    cmap = meta['cmap']
    input_h = 1024
    input_w = 2048
    classes = model_specs['classes']
    label_stride = model_specs['feat_stride']

    start_idx = args.start
    end_idx = args.end

    image_list = []
    idx = 0
    with open(args.file_list) as f:
        for item in f.readlines():
            idx += 1
            if idx < start_idx:
                continue

            if idx > end_idx:
                break

            item = item.strip()
            image_list.append(os.path.join(args.data_root, item))

    net_args, net_auxs = mxutil.load_params_from_file(args.weights)
    net = fcrna_model_a1(classes, label_stride, bootstrapping=True)
    if net is None:
        raise NotImplementedError('Unknown network')
    contexts = [mx.gpu(int(_)) for _ in args.gpus.split(',')]
    mod = mx.mod.Module(net, context=contexts)

    crop_size = 2048
    save_dir = args.output

    x_num = len(image_list)

    transformers = [ts.Scale(crop_size, Image.CUBIC, False)]
    transformers += _get_transformer_image()
    transformer = ts.Compose(transformers)

    start = time.time()

    for i in range(x_num):
        time1 = time.time()

        sample_name = osp.splitext(osp.basename(image_list[i]))[0]
        out_path = osp.join(save_dir, '{}.png'.format(sample_name))
        if os.path.exists(out_path):
            continue

        im_path = osp.join(args.data_root, image_list[i])
        rim = np.array(Image.open(im_path).convert('RGB'), np.uint8)

        h, w = rim.shape[:2]
        need_resize = False
        if h != input_h or w != input_w:
            need_resize = True
            im = np.array(Image.fromarray(rim.astype(np.uint8, copy=False)).resize((input_w, input_h), Image.NEAREST))
        else:
            im = rim
        im = transformer(im)
        imh, imw = im.shape[:2]

        # init
        label_h, label_w = input_h / label_stride, input_w / label_stride
        test_steps = 1
        pred_stride = label_stride / test_steps
        pred_h, pred_w = label_h * test_steps, label_w * test_steps

        input_data = np.zeros((1, 3, input_h, input_w), np.single)
        input_label = 255 * np.ones((1, label_h * label_w), np.single)
        dataiter = mx.io.NDArrayIter(input_data, input_label)
        batch = dataiter.next()
        mod.bind(dataiter.provide_data, dataiter.provide_label, for_training=False, force_rebind=True)
        if not mod.params_initialized:
            mod.init_params(arg_params=net_args, aux_params=net_auxs)

        nim = np.zeros((3, imh + label_stride, imw + label_stride), np.single)
        sy = sx = label_stride / 2
        nim[:, sy:sy + imh, sx:sx + imw] = im.transpose(2, 0, 1)

        net_preds = np.zeros((classes, pred_h, pred_w), np.single)
        # sy = sx = pred_stride // 2 + np.arange(test_steps) * pred_stride
        # sy = sx = sy[0]
        input_data = np.zeros((1, 3, input_h, input_w), np.single)
        input_data[0, :, :imh, :imw] = nim[:, sy:sy + imh, sx:sx + imw]
        batch.data[0] = mx.nd.array(input_data)
        mod.forward(batch, is_train=False)
        this_call_preds = mod.get_outputs()[0].asnumpy()[0]
        if args.test_flipping:
            batch.data[0] = mx.nd.array(input_data[:, :, :, ::-1])
            mod.forward(batch, is_train=False)
            this_call_preds = 0.5 * (this_call_preds + mod.get_outputs()[0].asnumpy()[0][:, :, ::-1])
        net_preds[:, 0:0 + pred_h:test_steps, 0:0 + pred_w:test_steps] = this_call_preds

        # compute pixel-wise predictions
        interp_preds = interp_preds_as(rim.shape[:2], net_preds, pred_stride, imh, imw)
        pred_label = interp_preds.argmax(0)
        if id_2_label is not None:
            pred_label = id_2_label[pred_label]

        # save predicted labels into an image
        im_to_save = Image.fromarray(pred_label.astype(np.uint8))
        if cmap is not None:
            im_to_save.putpalette(cmap.ravel())

        if need_resize:
            im_to_save = im_to_save.resize((w, h), Image.NEAREST)

        im_to_save.save(out_path)

        time2 = time.time()
        print("{}/{} {} finish in {} s\n".format(i, x_num, out_path, time2 - time1))

    logger.info('Done in %.2f s.', time.time() - start)


if __name__ == "__main__":
    util.cfg['choose_interpolation_method'] = True

    args, model_specs = parse_args()

    logger = util.set_logger(args.output, args.log_file)
    logger.info('start with arguments %s', args)
    logger.info('and model specs %s', model_specs)

    do(args, model_specs, logger)

