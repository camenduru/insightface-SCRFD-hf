#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import pathlib
import subprocess
import sys
import urllib.request

if os.environ.get('SYSTEM') == 'spaces':
    import mim
    mim.install('mmcv-full==1.3.3', is_yes=True)

    subprocess.call('pip uninstall -y opencv-python'.split())
    subprocess.call('pip uninstall -y opencv-python-headless'.split())
    subprocess.call('pip install opencv-python-headless==4.5.5.64'.split())
    subprocess.call('pip install terminaltables==3.1.0'.split())
    subprocess.call('pip install mmpycocotools==12.0.3'.split())

    subprocess.call('pip install insightface==0.6.2'.split())


import cv2
import gradio as gr
import huggingface_hub
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'insightface/detection/scrfd')

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

REPO_URL = 'https://github.com/deepinsight/insightface/tree/master/detection/scrfd'
TITLE = 'insightface Face Detection (SCRFD)'
DESCRIPTION = f'This is a demo for {REPO_URL}.'
ARTICLE = None

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--face-score-slider-step', type=float, default=0.05)
    parser.add_argument('--face-score-threshold', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    return parser.parse_args()


def load_model(model_size: str, device) -> nn.Module:
    ckpt_path = huggingface_hub.hf_hub_download(
        'hysts/insightface',
        f'models/scrfd_{model_size}/model.pth',
        use_auth_token=TOKEN)
    scrfd_dir = 'insightface/detection/scrfd'
    config_path = f'{scrfd_dir}/configs/scrfd/scrfd_{model_size}.py'
    model = init_detector(config_path, ckpt_path, device.type)
    return model


def update_test_pipeline(model: nn.Module, mode: int):
    cfg = model.cfg
    pipelines = cfg.data.test.pipeline
    for pipeline in pipelines:
        if pipeline.type == 'MultiScaleFlipAug':
            if mode == 0:  #640 scale
                pipeline.img_scale = (640, 640)
                if hasattr(pipeline, 'scale_factor'):
                    del pipeline.scale_factor
            elif mode == 1:  #for single scale in other pages
                pipeline.img_scale = (1100, 1650)
                if hasattr(pipeline, 'scale_factor'):
                    del pipeline.scale_factor
            elif mode == 2:  #original scale
                pipeline.img_scale = None
                pipeline.scale_factor = 1.0
            transforms = pipeline.transforms
            for transform in transforms:
                if transform.type == 'Pad':
                    if mode != 2:
                        transform.size = pipeline.img_scale
                        if hasattr(transform, 'size_divisor'):
                            del transform.size_divisor
                    else:
                        transform.size = None
                        transform.size_divisor = 32


def detect(image: np.ndarray, model_size: str, mode: int,
           face_score_threshold: float,
           detectors: dict[str, nn.Module]) -> np.ndarray:
    model = detectors[model_size]
    update_test_pipeline(model, mode)

    # RGB -> BGR
    image = image[:, :, ::-1]
    preds = inference_detector(model, image)
    boxes = preds[0]

    res = image.copy()
    for box in boxes:
        box, score = box[:4], box[4]
        if score < face_score_threshold:
            continue
        box = np.round(box).astype(int)

        line_width = max(2, int(3 * (box[2:] - box[:2]).max() / 256))
        cv2.rectangle(res, tuple(box[:2]), tuple(box[2:]), (0, 255, 0),
                      line_width)

    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    return res


def main():
    gr.close_all()

    args = parse_args()
    device = torch.device(args.device)

    model_sizes = [
        '500m',
        '1g',
        '2.5g',
        '10g',
        '34g',
    ]
    detectors = {
        model_size: load_model(model_size, device=device)
        for model_size in model_sizes
    }
    modes = [
        '(640, 640)',
        '(1100, 1650)',
        'original',
    ]

    func = functools.partial(detect, detectors=detectors)
    func = functools.update_wrapper(func, detect)

    image_path = pathlib.Path('selfie.jpg')
    if not image_path.exists():
        url = 'https://raw.githubusercontent.com/peiyunh/tiny/master/data/demo/selfie.jpg'
        urllib.request.urlretrieve(url, image_path)
    examples = [[image_path.as_posix(), '10g', modes[0], 0.3]]

    gr.Interface(
        func,
        [
            gr.inputs.Image(type='numpy', label='Input'),
            gr.inputs.Radio(
                model_sizes, type='value', default='10g', label='Model'),
            gr.inputs.Radio(
                modes, type='index', default=modes[0], label='Mode'),
            gr.inputs.Slider(0,
                             1,
                             step=args.face_score_slider_step,
                             default=args.face_score_threshold,
                             label='Face Score Threshold'),
        ],
        gr.outputs.Image(type='numpy', label='Output'),
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
