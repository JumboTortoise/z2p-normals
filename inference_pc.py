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
FRAME_DIRECTORY = Path(__file__).resolve().absolute().parent.joinpath('frames')
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


def load_metadata():  # load metadata from the .json file
    with open(ROOT_PATH / 'models' / 'default_settings.json', 'r') as file:
        models_meta = json.load(file)
    return models_meta


def load_model_from_checkpoint(opts):
    device = torch.device('cpu')
    #models_meta = load_metadata()
    #if opts.model_type not in models_meta.keys():
    #    raise ValueError(f'no model type {opts.model_type}')

    #num_controls = models_meta[opts.model_type]['len_style']
    #num_controls = models_meta["regular"]['len_style']
    model = PosADANet(1, 3, padding=opts.padding, bilinear=not opts.trans_conv,
        nfreq=opts.nfreq, magnitude=opts.freq_magnitude).to(device)
    model.load_state_dict(torch.load(opts.checkpoint, map_location=device))
    return model

def clear_frames():
    files = [pth.resolve() for pth in FRAME_DIRECTORY.iterdir() if pth.is_file() and file.suffix.endswith('png')]
    for pth in files:
        pth.unlink()

def video(opts,frames=128):
    timer = util.timer_factory()
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    clear_frames()
    zbuffers = []
    with timer('load pc'):
        pc = torch.tensor(np.load(opts.pc)).float()
        center = render_util.center_of_mass(pc)
        for f in range(len(frames)):
            pc = render_util.rotate_pc(pc,0,0,0.3,pivot=center)
            zbuffer = parse_pts(get_image_from_point_cloud(pc,opts.focal_length,opts.zbuffer_height,opts.zbuffer_width),radius=opts.splat_size)
            #zbuffer = zbuffer[opts.width_ranges[0]: opts.width_ranges[1], opts.height_ranges[0]:opts.height_ranges[1]]
            zbuffer = resize(zbuffer,target=[opts.sampled_width,opts.sampled_height])

            if opts.flip_z:
                zbuffer = np.flip(zbuffer, axis=0).copy()
            zbuffer: torch.Tensor = torch.from_numpy(zbuffer).float().to(device)

            zbuffer = zbuffer.unsqueeze(-1).permute(2, 0, 1)
            zbuffer: torch.Tensor = zbuffer.float().to(device).unsqueeze(0)

            zbuffers.append(zbuffer)

    model = load_model_from_checkpoint(opts).to(device)

    model.eval()
    f = 0
    while f < frames:
        
        
        batch = torch.concat(zbuffers[f:f + opts.video_batch],dim=0)
        
        with torch.no_grad():
            generated = model(batch.float())

        for i in range(len(generated)):
            frame = generated[i]
            im = frame.permute(1,2,0).clip(0,1)*255
            im = im.detach().cpu().numpy()
            cv2.imwrite(FRAME_DIRECTORY / f"frame_{f + i}.png",im)
        f += len(batch)
    
    fps = 30
    frame_size = (1920, 1080)  # Adjust to your frame dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec (may vary based on your system)
    video_writer = cv2.VideoWriter(output_video_filename, fourcc, fps, frame_size)

    # Loop through PNG files and add them to the video
    for png_file in png_files:
        frame_path = os.path.join(input_directory, png_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            video_writer.write(frame)

def single(opts):
    timer = util.timer_factory()
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))

    with timer('load pc'):
        pc = torch.tensor(np.load(opts.pc)).float()
        center = render_util.center_of_mass(pc)
        pc = render_util.rotate_pc(pc, opts.rx, opts.ry, opts.rz,pivot=center)
        zbuffer = parse_pts(get_image_from_point_cloud(pc,opts.focal_length,opts.zbuffer_height,opts.zbuffer_width),radius=opts.splat_size)
        zbuffer = zbuffer[opts.height_ranges[0]: opts.height_ranges[1], opts.width_ranges[0]:opts.width_ranges[1]]
        zbuffer = resize(zbuffer,target=[opts.sampled_width,opts.sampled_height])

    if opts.flip_z:
        zbuffer = np.flip(zbuffer, axis=0).copy()

    if opts.show_results:
        plt.title("z-buffer")
        plt.imshow(zbuffer,cmap='gray')
        plt.show()
    zbuffer: torch.Tensor = torch.from_numpy(zbuffer).float().to(device)

    zbuffer = zbuffer.unsqueeze(-1).permute(2, 0, 1)
    zbuffer: torch.Tensor = zbuffer.float().to(device).unsqueeze(0)
    
    # creates export directory (if it does not exist) and zbuffer
    export_results_flag = opts.export_dir is not None 
    if export_results_flag:
        opts.export_dir.mkdir(exist_ok=True, parents=True)
        export_results(opts, [f'zbuffer'], zbuffer.detach())

    model = load_model_from_checkpoint(opts).to(device)

    model.eval()

    with torch.no_grad():
        generated = model(zbuffer.float())

    if opts.show_results:
        im = generated[0].permute(1,2,0).clip(0,1)*255
        im = im.detach().cpu().numpy().astype(np.uint8)
        
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # for some reason, output is in BGR? 
        
        """
        fig,axis = plt.subplots(nrows=1,ncols=3)
        axis[0].set_title('red')
        axis[0].imshow(im[:,:,0])
        axis[1].set_title('green')
        axis[1].imshow(im[:,:,1])
        axis[2].set_title('blue')
        axis[2].imshow(im[:,:,-1])
        plt.show()
        """
        
        plt.title("generated")
        plt.imshow(im)
        plt.show()

        if opts.matcap_comparison:
            apply_matcaps_to_view(opts, zbuffer, im)

        if opts.matcap is not None:
            
            matcap = apply_matcap_unnormalized(im,Image.open(opts.matcap))
            plt.title("matcap")
            plt.imshow(matcap)
            plt.show()
    
    if export_results_flag:
        export_results(opts, [f'rendered'], generated)

    print('done')

def apply_matcaps_to_view(opts,z_buffer, generated):
    view_dir = opts.pc.parent

    normals_file = view_dir / 'normals.png'

    normals = Image.open(normals_file)
    normals = np.array(normals)[:, :, :3]

    # Create a figure with subplots
    fig, axs = plt.subplots(1 + len(opts.matcap_comparison), 2, figsize=(12, 6))  # Adjust the figsize as needed

    axs[0,0].imshow(normals)
    axs[0,0].axis('off')  # Hide axis for the first image
    axs[0,0,].set_title('true normals')

    axs[1,0].imshow(z_buffer.squeeze(), cmap='gray')
    axs[1,0].axis('off')  # Hide axis for the second image
    axs[1,0].set_title('z_buffer')

    for i,mat in enumerate(opts.matcap_comparison):
        matcap = Image.open(mat)

        gen_mat = matcap #apply_matcap_unnormalized(generated, matcap)
        true_mat = apply_matcap_unnormalized(normals, matcap)

        axs[i+1,0].imshow(gen_mat)
        axs[i+1,0].axis('off')  # Hide axis for additional images
        axs[i+1,0].set_title(f'generated image after matcap')

        axs[i+1,1].imshow(true_mat)
        axs[i+1,1].axis('off')  # Hide axis for additional images
        axs[i+1,1].set_title(f'true image after matcap')


    # Adjust the layout for better spacing between subplots
    plt.tight_layout()

    # Show the figure
    plt.show()
    





if __name__ == '__main__':
    sample_command = "python inference_pc.py --show_results --pc /path/to/cloud.npy --checkpoint /path/to/model.pt --export_dir /where/to/save"
    parser = argparse.ArgumentParser(
        prog='get single frame or video visualization of point cloud',
        epilog=f'example command: {"python inference_pc.py " + sample_command}')
    parser.add_argument('--export_dir', type=Path, default=None, required=False,
                        help='path to export directory, if blank dont save')

    parser.add_argument('--focal_length',type=int,default=50)
    parser.add_argument('--zbuffer_width',type=int,default=960)
    parser.add_argument('--zbuffer_height',type=int,default=540)
    parser.add_argument('--width_ranges',type=int,nargs=2,default=[100, 960])
    parser.add_argument('--height_ranges',type=int,nargs=2,default=[0, 540])
    parser.add_argument('--sampled_width',type=int,default=500)
    parser.add_argument('--sampled_height',type=int,default=-1)

    parser.add_argument('--pc', type=Path, required=True, help='path to input point cloud to visualize')
    parser.add_argument('--trans_conv', action='store_true',
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
    parser.add_argument('--flip_z', action='store_true', help='flip the z axis')
    parser.add_argument('--dy', default=290, type=int, required=False,
                        help='translation of the input pc in the vertical direction of the image')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='path to the .pt model checkpoint,'
                             ' if None a pretrained model will be downloaded from gdrive according to --model_type')
    parser.add_argument('--matcap',type=Path,default=None,required=False,
                        help='a matcap to apply to rendered normal map')
    parser.add_argument('--show_results', action='store_true', help='show results with matplotlib')
    parser.add_argument('--splat_size',type=int,default=1)
    parser.add_argument('--nfreq', type=int, default=20)
    parser.add_argument('--freq_magnitude', type=int, default=10)
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--matcap_comparison', type = Path, nargs='+')
    parser.add_argument('--video_batch',type=int,default=4)


    command = f"--show_results --pc C:/data_set/chair/rotation_1/cloud_0.npy --checkpoint C:/z2p_normals/epoch_9u.pt --export_dir C:/z2p_normals/models --matcap_comparison C:/z2p_normals/z2p-normals/matcaps/basic_side.png C:/z2p_normals/z2p-normals/matcaps/pearl.png"
    command_args = shlex.split(command)

# Parse the arguments
    opts = parser.parse_args(command_args)


    if opts.video:
        video(opts)
    else:
        single(opts)

    
