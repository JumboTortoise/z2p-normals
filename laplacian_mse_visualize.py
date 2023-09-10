import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pathlib

torch.backends.cudnn.deterministic = True
torch.manual_seed(42)

laplacian_kernel = np.array([
        [0,-1,0],
        [-1,4,-1],
        [0,-1,0]
    ])

gaussian_kernel = np.array([
    [1/16,1/8,1/16],
    [1/8,1/4,1/8],
    [1/16,1/8,1/16]
    ])

def main(args):
    if not args.normal_map.exists() or not args.normal_map.is_file():
        print("The given path does not point to a valid image")
        return

    try:
        img = np.array(Image.open(args.normal_map).convert("RGB"))
    except Exception as e:
        print("failed to load image:",e)
        return

    original_img = img
    print("input image shape:",img.shape)
    img = img[:,:,:2] # remove blue channel (alpha)
    img = (torch.tensor(img) / 255).permute(2,0,1).unsqueeze(0) # to torch tensor, normalize, change axis order and add minibatch
    img.requires_grad = False

    # kernel into a pytorch tensor
    kernel = torch.tensor(laplacian_kernel).unsqueeze(0).unsqueeze(0).repeat(2,1,1,1).float()
    gkernel = torch.tensor(gaussian_kernel).unsqueeze(0).unsqueeze(0).repeat(2,1,1,1).float()
    print("gkernel size:",gkernel.shape,"kernel size:",kernel.shape)
    print("input tensor shape(pytorch):",img.shape)
    print("applying convolution...")
    
    scale = 2
    result = torch.abs(F.conv2d(img, kernel, bias=None, stride=1, padding=1, dilation=1, groups=2))*scale
    print(f"max:{torch.max(result)},min:{torch.min(result)}")
    result = torch.abs(F.conv2d(result,gkernel,bias=None,stride=1,padding=1,dilation=1,groups=2))
    print("output tensor shape:",result.shape)
    
    print("plotting results...")

    fig,axes = plt.subplots(nrows=1,ncols=3)
    fig.suptitle("Convolution results")
    axes[0].set_title("Original")
    axes[0].imshow(original_img)
    axes[1].set_title("Red(X axis normals)")
    axes[1].imshow(result[0,0,:,:],cmap='gray')
    axes[2].set_title("Green(Y axis normals)")
    axes[2].imshow(result[0,1,:,:],cmap='gray')
    plt.show()

    print("done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="visualize the laplacian filter on a normal map")
    parser.add_argument("normal_map",type=pathlib.Path,help='path to the normal map to visualize')
    main(parser.parse_args())


