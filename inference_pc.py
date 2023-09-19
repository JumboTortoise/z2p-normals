import argparse
import json
from pathlib import Path
import shlex
from typing import List

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

# used to create matcap
from matcap import apply_matcap_unnormalized

# used to create point cloud
from convert_point_cloud import get_image_from_point_cloud
from data import parse_pts,resize


import render_util
import util
from models import PosADANet
from PIL import Image

ROOT_PATH = Path(__file__).resolve().absolute().parent
torch.manual_seed(10)


def export_results(opts, names: List[str], generated: torch.Tensor):
    # saves a result with the given name to the export directory
    for i, name in enumerate(names):
        name = name.replace('.npy', '')
        name = f'{name}.png'

        export_path = opts.export_dir / name
        o_img = generated[i].permute(1, 2, 0) # from pytorch image layout to standard image layout

        o_img = o_img.cpu().numpy() * 255
        cv2.imwrite(str(export_path), o_img)


def load_model_from_checkpoint(opts):
    device = torch.device('cpu')
    model = PosADANet(1, 3, padding=opts.padding, bilinear=not opts.trans_conv,
        nfreq=opts.nfreq, magnitude=opts.freq_magnitude).to(device)
    model.load_state_dict(torch.load(opts.checkpoint, map_location=device))
    return model

def single(opts):
    timer = util.timer_factory()
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))

    with timer('load pc'):
        pc = torch.tensor(np.load(opts.pc)).float()
        center = render_util.center_of_mass(pc)
        pc = render_util.rotate_pc(pc, opts.rx, opts.ry, opts.rz,pivot=center)
        pc = render_util.scale_pc(pc,opts.scale,pivot=center)
        zbuffer = parse_pts(get_image_from_point_cloud(pc,opts.focal_length,opts.zbuffer_height,opts.zbuffer_width),radius=opts.splat_size)
        zbuffer = zbuffer[opts.height_ranges[0]: opts.height_ranges[1], opts.width_ranges[0]:opts.width_ranges[1]]
        zbuffer = resize(zbuffer,target=[opts.sampled_width,opts.sampled_height])

    if opts.flip_z:
        zbuffer = np.flip(zbuffer, axis=0).copy()

    original_zbuffer = zbuffer
    zbuffer: torch.Tensor = torch.from_numpy(zbuffer).float().to(device)

    zbuffer = zbuffer.unsqueeze(-1).permute(2, 0, 1)
    zbuffer: torch.Tensor = zbuffer.float().to(device).unsqueeze(0)
    
    # creates export directory (if it does not exist) and zbuffer
    export_results_flag = opts.export_dir is not None 
    #if export_results_flag:
    #    opts.export_dir.mkdir(exist_ok=True, parents=True)
    #    export_results(opts, [f'zbuffer'], zbuffer.detach())

    model = load_model_from_checkpoint(opts).to(device)

    model.eval()

    with torch.no_grad():
        generated = model(zbuffer.float())

    im = generated[0].permute(1,2,0).clip(0,1)*255
    im = im.detach().cpu().numpy().astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  
    
    matcap = None
    if opts.matcap is not None:
        m = np.array(Image.open(opts.matcap).convert("RGB"))
        matcap = apply_matcap_unnormalized(im,m)

    # show the results
    if opts.show_results:
        plt.title("z-buffer")
        plt.imshow(original_zbuffer,cmap='gray')
        plt.show()
        plt.title("generated")
        plt.imshow(im)
        plt.show()
        if matcap is not None:
            plt.title("matcap")
            plt.imshow(matcap)
            plt.show()

    # export results
    if export_results_flag:
        export_results(opts, [f'zbuffer'], zbuffer)
        export_results(opts, [f'normal'], generated)
        if matcap is not None:
            export_results(opts, [f'rendered'], torch.tensor(cv2.cvtColor(matcap, cv2.COLOR_RGB2BGR)).permute(2,0,1).unsqueeze(0)/255)

    print('done')

if __name__ == '__main__':
    sample_command = "python inference_pc.py --show-results --pc /path/to/cloud.npy --checkpoint /path/to/model.pt --export-dir /where/to/save"
    parser = argparse.ArgumentParser(
        prog='get single frame or video visualization of point cloud',
        epilog=f'example command: {"python inference_pc.py " + sample_command}')
    parser.add_argument('--export-dir',dest='export_dir', type=Path, default=None, required=False,
                        help='path to export directory, if blank dont save')

    parser.add_argument('--focal-length',dest='focal_length',type=int,default=50)
    parser.add_argument('--zbuffer-width',dest='zbuffer_width',type=int,default=960)
    parser.add_argument('--zbuffer-height',dest='zbuffer_height',type=int,default=540)
    parser.add_argument('--width-ranges',dest='width_ranges',type=int,nargs=2,default=[100, 960])
    parser.add_argument('--height-ranges',dest='height_ranges',type=int,nargs=2,default=[0, 540])
    parser.add_argument('--sampled-width',dest='sampled_width',type=int,default=500)
    parser.add_argument('--sampled-height',dest='sampled_height',type=int,default=-1)

    parser.add_argument('--pc', type=Path, required=True, help='path to input point cloud to visualize')
    parser.add_argument('--trans-conv',dest='trans_conv', action='store_true',
                        help='use a model with transconv instead of bilinear upsampling')
    parser.add_argument('--padding', default='zeros', type=str, required=False, help='padding type for the model')
    parser.add_argument('--scale', default=1.0, type=float, required=False,
                        help='pc scale before the 2D z-buffer projection')
    parser.add_argument('--rx', default=-1.9, type=float, required=False,
                        help='rotation on the input pc around the x axis')
    parser.add_argument('--ry', default=0.5, type=float, required=False,
                        help='rotation on the input pc around the y axis')
    parser.add_argument('--rz', default=2.0, type=float, required=False,
                        help='rotation on the input pc around the z axis')
    parser.add_argument('--flip-z',dest='flip_z', action='store_true', help='flip the z axis')
    parser.add_argument('--dy', default=290, type=int, required=False,
                        help='translation of the input pc in the vertical direction of the image')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='path to the .pt model checkpoint,'
                             ' if None a pretrained model will be downloaded from gdrive according to --model_type')
    parser.add_argument('--matcap',type=Path,default=None,required=False,
                        help='a matcap to apply to rendered normal map')
    parser.add_argument('--show-results',dest='show_results', action='store_true', help='show results with matplotlib')
    parser.add_argument('--splat-size',dest='splat_size',type=int,default=1)
    parser.add_argument('--nfreq', type=int, default=20)
    parser.add_argument('--freq-magnitude',dest='freq_magnitude', type=int, default=10)


    opts = parser.parse_args()
    single(opts)