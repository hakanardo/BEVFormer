# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
import sys
sys.path.append('.')
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets import custom_build_dataset
# from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
#from tools.misc.fuse_conv_bn import fuse_module


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--samples', default=2000, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    print(cfg.data.test)
    dataset = custom_build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    #if args.fuse_conv_bn:
    #    model = fuse_module(model)

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with several samples and take the average
    for i, data in enumerate(data_loader):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        from vi3o.image import imwrite, ptpscale
        import cv2
        import numpy as np
        img = data['img'][0].data[0][0,0]
        meta = data['img_metas'][0].data[0][0]
        bbox = result[0]['pts_bbox']
        mask = bbox['scores_3d'] > 0.5
        bbox['labels_3d'][mask]
        pkt = bbox['boxes_3d'].center[mask]
        proj = (torch.hstack([pkt, torch.ones(pkt.shape[0], 1)]).numpy() @ meta['lidar2img'][0].T)
        mask2 = proj[:, 2] > 1e-3
        proj = proj[mask2]
        proj = proj[:, :2] / proj[:, 2:3]
        classes = [dataset.CLASSES[i] for i in bbox['labels_3d'][mask][mask2]]

        drw = ptpscale(img.detach().cpu().numpy().transpose([1, 2, 0]).copy())
        for (x, y), c in zip(proj, classes):
            col = {'pedestrian': (255,0,0), 'car': (0,255,0)}[c]
            cv2.circle(drw, (int(x), int(y)), 5, col, -1)
        imwrite(drw, "t.png")


        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {args.samples}], '
                      f'fps: {fps:.1f} img / s')

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} img / s')
            break


if __name__ == '__main__':
    main()
