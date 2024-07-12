# import flow_vis
import argparse
import cv2
import glob
import os
import shutil
import torch
from archs.realviformer_arch import RealViformer
from data_util import read_img_seq
from img_util import tensor2img
import torch.nn.functional as F

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
        outputs = outputs[...,padh*4:, padw*4:]
    outputs = outputs.squeeze(0)
    outputs = list(outputs)
    for output, imgname in zip(outputs, imgnames):
        output = tensor2img(output)
        cv2.imwrite(os.path.join(save_path, f'{imgname}.png'), output)


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
    parser.add_argument('--input_path', type=str, default='datasets/REDS4/sharp_bicubic/000', help='input test image folder')
    parser.add_argument('--save_path', type=str, default='results/BasicVSR', help='save image path')
    parser.add_argument('--interval', type=int, default=100, help='interval size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    model = RealViformer(num_feat=48,
                            num_blocks=[2,3,4,1],
                            spynet_path=None,
                            heads=[1,2,4],
                            ffn_expansion_factor=2.66,
                            merge_head=2,
                            bias=False,
                            LayerNorm_type='BiasFree',
                            ch_compress=True,
                            squeeze_factor=[4, 4, 4],
                            masked=True)
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

