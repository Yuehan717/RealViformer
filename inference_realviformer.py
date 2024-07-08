# import flow_vis
import argparse
from dataclasses import replace
import cv2
import glob
import os
import shutil
import torch
import yaml
import pickle
from collections import OrderedDict
import csv
from basicsr.archs.realviformer_arch import RealViformer
from basicsr.data.data_util import read_img_seq, read_flow_seq
from basicsr.utils.img_util import tensor2img
import matplotlib.pyplot as plt
import torch.nn.functional as F
# from basicsr.archs.pretrain_arch import PretrainVSR

def inference(imgs, imgnames, model, save_path):
    with torch.no_grad():
        padded = False
        h, w= imgs.shape[-2:]
        ah, aw = h%4, w%4
        padh = 0 if ah == 0 else 4-ah
        padw = 0 if aw == 0 else 4-aw
        if padh !=0 or padw !=0:
            padded = True
            # print(imgs.size())
            imgs = F.pad(imgs.squeeze(0), pad=(padw,0,padh,0), mode='reflect').unsqueeze(0)
        outputs = model(imgs)
    # save imgs
    if padded:
        outputs = outputs[...,padh:, padw:]
    outputs = outputs.squeeze(0)
    outputs = list(outputs)
    for output, imgname in zip(outputs, imgnames):
        output = tensor2img(output)
        cv2.imwrite(os.path.join(save_path, f'{imgname}.png'), output)


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def inference_vid(args, input_path, save_path, device, model, use_ffmpeg):
    # load data and inference
    imgs_list = sorted(glob.glob(os.path.join(input_path, '*')))
    num_imgs = len(imgs_list)
    if len(imgs_list) <= args.interval:  # too many images may cause CUDA out of memory
        imgs, imgnames = read_img_seq(imgs_list, return_imgname=True)
        imgs = imgs.unsqueeze(0).to(device)
        inference(imgs, imgnames, model, save_path)
    else:
        for idx in range(0, num_imgs, args.interval):
            interval = min(args.interval, num_imgs - idx)
            imgs, imgnames = read_img_seq(imgs_list[idx:idx + interval], return_imgname=True)
            imgs = imgs.unsqueeze(0).to(device)
            inference(imgs, imgnames, model, save_path)
            print(f"{idx}/{num_imgs}")

    # delete ffmpeg output images
    if use_ffmpeg:
        shutil.rmtree(input_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/BasicVSR_REDS4.pth')
    parser.add_argument(
        '--input_path', type=str, default='datasets/REDS4/sharp_bicubic/000', help='input test image folder')
    parser.add_argument('--save_path', type=str, default='results/BasicVSR', help='save image path')
    parser.add_argument('--interval', type=int, default=100, help='interval size')
    parser.add_argument('--opt', type=str, default='experiments/DynamicVSR_SeparateBlock_flow_pfeature_Tonly/train_DynamicVSR_SeparateBlock_REDS.yml')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    # set up model
    params = opt['network_g']
    num_feat = params.get('num_feat', 48)
    num_blocks = params.get('num_blocks', [])
    spynet_path = params.get('spynet_path', 'experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth')
    heads = params.get('heads', [])
    ffn_expansion_factor = params.get('ffn_expansion_factor', 2.66)
    masked = params.get('masked', False)
    merge_masked= params.get('merge_masked',False)
    merge_compress= params.get('merge_compress',False)
    merge_compress_factor= params.get('merge_compress_factor',1.0)
    bias= params.get('bias',False)
    LayerNorm_type= params.get('LayerNorm_type','BiasFree')
    ch_compress= params.get('ch_compress',False)
    squeeze_factor= params.get('squeeze_factor',[1,1,1])
    merge_head = params.get('merge_head',1)
    model = RealViformer(num_feat=num_feat, num_blocks=num_blocks, spynet_path=spynet_path, heads=heads, ffn_expansion_factor=ffn_expansion_factor, 
                            masked=masked,merge_head=merge_head, merge_masked=merge_masked, merge_compress=merge_compress, merge_compress_factor=merge_compress_factor,
                            bias=bias, LayerNorm_type=LayerNorm_type, ch_compress=ch_compress, squeeze_factor=squeeze_factor)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=False)
    model.eval()
    model = model.to(device)

    os.makedirs(args.save_path, exist_ok=True)

    # extract images from video format files
    input_path = args.input_path
    use_ffmpeg = False
    if not os.path.isdir(input_path):
        use_ffmpeg = True
        video_name = os.path.splitext(os.path.split(args.input_path)[-1])[0]
        input_path = os.path.join('./BasicVSR_tmp', video_name)
        os.makedirs(os.path.join('./BasicVSR_tmp', video_name), exist_ok=True)
        os.system(f'ffmpeg -i {args.input_path} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {input_path} /frame%08d.png')
        
        imgs_list = sorted(glob.glob(os.path.join(input_path, '*')))
        num_imgs = len(imgs_list)
        if len(imgs_list) <= args.interval:  # too many images may cause CUDA out of memory
            imgs, imgnames = read_img_seq(imgs_list, return_imgname=True)
            imgs = imgs.unsqueeze(0).to(device)
            inference(imgs, imgnames, model, args.save_path)
        else:
            for idx in range(0, num_imgs, args.interval):
                interval = min(args.interval, num_imgs - idx)
                imgs, imgnames = read_img_seq(imgs_list[idx:idx + interval], return_imgname=True)
                imgs = imgs.unsqueeze(0).to(device)
                inference(imgs, imgnames, model, args.save_path)
                print(f"{idx}/{num_imgs}")

        # delete ffmpeg output images
        if use_ffmpeg:
            shutil.rmtree(input_path)
    else:
        ### input_path is a dictionary
        video_names = os.listdir(input_path)
        # print(video_names[0])
        if video_names[0].endswith('.png'):
            inference_vid(args, input_path, args.save_path, device, model, use_ffmpeg)
        else:
            video_names.sort()
            video_names = video_names[:44]
            for video_name in video_names:
                video_path = os.path.join(input_path, video_name)
                save_path = os.path.join(args.save_path, video_name)
                os.makedirs(save_path, exist_ok=True)
                # print(save_path)
                inference_vid(args, video_path, save_path, device, model, use_ffmpeg)


if __name__ == '__main__':
    main()

