import argparse
from inspect import getmembers, isfunction
from pathlib import Path,PurePath

import cv2 as cv
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

import data as data
import losses
import util
from models import PosADANet

import matplotlib.pyplot as plt
import json

losses_funcs = {}
for val in getmembers(losses):
    if isfunction(val[1]):
        losses_funcs[val[0]] = val[1]


def log_images(path, msg, img_tensor, style = None):
    # img_tensor = util.embed_color(img_tensor, style[:, :3])
    # white_img = util.embed_background(img_tensor)
    for i, img in enumerate(img_tensor):
        img = img.permute(1, 2, 0)
        img = img.clip(0, 1) * 255
        img = img.detach().cpu().numpy()
        cv.imwrite(f'{str(path)}/{msg}_{i}.png', img)


def get_loss_function(lname):
    return losses_funcs[lname]

class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, PurePath):
            return str(obj)
        return json.JSONEncoder.default(obj)

def train(opts):
    run_name =" trial train"
    train_export_dir = opts.export_dir
    
    if not opts.test:
        json_content = json.dumps(vars(opts),cls = PathEncoder,indent=4)
        with (train_export_dir / "settings.json").open('w') as f:
            f.write(json_content)

    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))

    # mixed precision
    scaler = None
    using_mixed_precision = False
    if torch.cuda.is_available() and opts.mixed_precision and not opts.test:
        using_mixed_precision = True
        scaler = GradScaler()

    train_set = data.GenericDataset(opts.data, splat_size=opts.splat_size, cache=opts.cache)
    
    train_loader = DataLoader(train_set, batch_size=opts.batch_size,
                              shuffle=True, num_workers=opts.num_workers, pin_memory=True)
    

    num_samples = len(train_loader)
    model = PosADANet(input_channels=1, output_channels=3,
                      padding=opts.padding, bilinear=not opts.trans_conv,
                      nfreq=opts.nfreq, magnitude=opts.freq_magnitude).to(device)

    if opts.checkpoint is not None:
        model.load_state_dict(torch.load(opts.checkpoint))

    lfuncs = {lname:get_loss_function(lname) for lname in opts.losses}

    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    global_step = 0
    avg_loss = util.RunningAverage()
    avg_test_loss = util.RunningAverage()
    for epoch in range(opts.start_epoch, opts.epochs):
        print(f"starting epoch {epoch}")
        avg_loss.reset()
        model.train()
        for i, (img, zbuffer) in enumerate(train_loader):
            optimizer.zero_grad()
            
            #print("train:",img.shape,zbuffer.shape)
            img: torch.Tensor = img.float().to(device)
            zbuffer: torch.Tensor = zbuffer.float().to(device)
            
            if using_mixed_precision: # using mixed precision
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    generated = model(zbuffer.float())
                    loss = 0
                    for weight, lname in zip(opts.l_weight, opts.losses):
                        loss += weight * lfuncs[lname](generated, img)

                    if not opts.test and loss != 0:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
            else:
                generated = model(zbuffer.float())
                loss = 0
                for weight, lname in zip(opts.l_weight, opts.losses):
                    loss += weight * lfuncs[lname](generated, img)

                if not opts.test and loss != 0:
                    loss.backward()
                    optimizer.step()
            generated = generated.float()

            if global_step % opts.log_iter == 0 or global_step == 50:
                to_display = 4
                expanded_z_buffer = zbuffer.repeat((1, 3, 1, 1))

                if generated.shape[0] > to_display:
                    generated = generated[:to_display,:,:,:]
                    expanded_z_buffer = expanded_z_buffer[:to_display,:,:,:]
                    img = img[:to_display,:,:,:]
                
                cat_img = torch.cat([img, generated, expanded_z_buffer.clamp(0, 1)], dim=2)

                log_images(train_export_dir, f'train_imgs{global_step}', cat_img.detach())

            global_step += 1

            avg_loss.add(loss.item())
            print(f'{run_name}; epoch: {epoch}; iter: {i}/{num_samples} loss: {loss}')

        """
        region test
        model.eval()
        avg_test_loss.reset()
        for i, (img, zbuffer, color) in enumerate(test_loader):
            with torch.no_grad():
                zbuffer = zbuffer.float().to(device)
                color = color.float().to(device)
                img = img.float().to(device)
                generated = model(zbuffer.float(), color)
                test_loss = 0
                for weight, lname in zip(opts.l_weight, opts.losses):
                    test_loss += weight * get_loss_function(lname)(generated, img)
                avg_test_loss.add(test_loss.item())

                expanded_z_buffer = zbuffer.repeat((1, 4, 1, 1))
                expanded_z_buffer[:, -1, :, :] = 1
                cat_img = torch.cat([img, generated, expanded_z_buffer.clamp(0, 1)], dim=2)
                log_images(test_export_dir, f'pairs_epoch_{epoch}', cat_img.detach(), color)
        """
        #endregion
        print(f'average loss: {avg_loss.get_average()}')
        
        if not opts.test:
            torch.save(model.state_dict(), opts.export_dir / f'epoch:{epoch}.pt')


if __name__ == '__main__':
    command = r"""--data /path/to/dataset --export_dir /where/to/save/checkpoint --batch_size 4 --num_workers 4 --epochs 10 --log_iter 1000 --losses masked_mse intensity masked_pixel_intensity --l_weight 1 0.7 1 --splat_size 1"""

    parser = argparse.ArgumentParser(
                    prog='python train.py',
                    description='This is the training script for z2p-normals, a version of z2p made to visualize normal maps from point clouds',
                    epilog=f'example command: {"python train.py " + command}')
    parser.add_argument('--data', type=Path,help='path to the dataset root directory')
    parser.add_argument('--export_dir', type=Path,help='path to the directory where checkpoints and logs will be kept')
    parser.add_argument('--test', action='store_true',help='if present, the checkpoint will be tested and not trained')
    parser.add_argument('--checkpoint', type=Path, default=None, help='path to checkpoint, use to continue training from checkpoint')
    parser.add_argument('--start_epoch', type=int, default=0,help='epoch to start with, should only be larger then 0 when resuming to train from checkpoint')
    parser.add_argument('--batch_size', type=int,help='batch size')
    parser.add_argument('--num_workers', type=int,help='data loader workers')
    parser.add_argument('--nfreq', type=int, default=20,help='number of fourier feature frequencies')
    parser.add_argument('--freq_magnitude', type=int, default=10,help='size of the fourier features frequencies range')
    parser.add_argument('--log_iter', type=int, default=1000,help='the number of iterations to wait before logging new sample images')
    parser.add_argument('--epochs', type=int,help='number of epochs to train/test')
    parser.add_argument('--lr', type=float, default=3e-4,help='learning rate')
    parser.add_argument('--losses', nargs='+', default=['mse', 'intensity'],help='the types of losses that should be used for training/testing')
    parser.add_argument('--l_weight', nargs='+', default=[1, 1], type=float,help='the weights that should be applied to each loss function, ordered accourding to --losses')
    parser.add_argument('--mixed_precision',action='store_true',help='train in mixed precision mode') 
    parser.add_argument('--padding', default='zeros', type=str)
    parser.add_argument('--trans_conv', action='store_true',help='use transposed convolution instead of bilinear upsampling')
    parser.add_argument('--cache', action='store_true',help='cache the z-buffers for faster loading(requires a lot of storage)')
    parser.add_argument('--splat_size', type=int,default=1,help='splat size for z-buffer')
    
    opts = parser.parse_args()
    if opts.test:
        with torch.no_grad():
            train(opts)
    else:
        train(opts)
