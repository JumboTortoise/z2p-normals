import argparse
from inspect import getmembers, isfunction
from pathlib import Path

import cv2 as cv
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

import data as data
import losses
import util
from models import PosADANet

import matplotlib.pyplot as plt

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


def train(opts):
    run_name =" trial train"
    train_export_dir = opts.export_dir

    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))

    # mixed precision
    scaler = None
    using_mixed_precision = False
    if torch.cuda.is_available() and opts.mixed_precision:
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

                    if loss != 0:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
            else:
                generated = model(zbuffer.float())
                loss = 0
                for weight, lname in zip(opts.l_weight, opts.losses):
                    loss += weight * lfuncs[lname](generated, img)

                if loss != 0:
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
        
        print(f'average train loss: {avg_loss.get_average()}')

        torch.save(model.state_dict(), opts.export_dir / f'epoch:{epoch}.pt')


if __name__ == '__main__':
    command = r"""--data /path/to/dataset --export_dir /where/to/save/checkpoint --batch_size 4 --num_workers 4 --epochs 10 --log_iter 1000 --losses masked_mse intensity masked_pixel_intensity --l_weight 1 0.7 1 --splat_size 1"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path)
    parser.add_argument('--export_dir', type=Path)
    #parser.add_argument('--test_data', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int)
    #parser.add_argument('--test_batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--nfreq', type=int, default=20)
    parser.add_argument('--freq_magnitude', type=int, default=10)
    parser.add_argument('--log_iter', type=int, default=1000)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--losses', nargs='+', default=['mse', 'intensity'])
    parser.add_argument('--l_weight', nargs='+', default=[1, 1], type=float)
    parser.add_argument('--mixed_precision',action='store_true') # added mixed precision support
    parser.add_argument('--tb', action='store_true')
    parser.add_argument('--padding', default='zeros', type=str)
    parser.add_argument('--trans_conv', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--splat_size', type=int,default=1)
    
    command = r"--data C:\data_set --export_dir C:\z2p_normals\models --batch_size 4 --num_workers 4 --epochs 10 --log_iter 1000 --losses masked_mse intensity masked_pixel_intensity --l_weight 1 0.7 1 --splat_size 3"


    #train(parser.parse_args(command.split(" ")))
    train(parser.parse_args())
