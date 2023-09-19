import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import time

THRESHOLD = 1
FORGROUND_VALUE = 255
BACKGROUND_VALUE = 0

def create_alpha(image):
    """adds the alpha channel into the blue channel, only required if not included in the dataset"""
    image = image.convert("RGB")
    # red and green contain normal, blue does not contain data
    red, green, blue = image.split()

    red,green = np.array(red),np.array(green)
    blue = np.zeros_like(red)
    blue[np.logical_or(red > THRESHOLD,blue > THRESHOLD)] = FORGROUND_VALUE
    blue[np.logical_and(red <= THRESHOLD,blue <= THRESHOLD)] = BACKGROUND_VALUE
    return Image.fromarray(np.stack((red,green,blue),axis=2),mode='RGB')



def apply_matcap_unnormalized(result,matcap):
    """
    NOTE: the matcap must be an RGB image, with no alpha channel
    
    result is assumed to be the output of the network, that is, an RGB image with 2 channels of normals
    and 1 alpha channel (the blue channel). The normals will be normalized to between -1 and 1, and the alpha 
    between -1 and 1.
    
    This function applies the given matcap to the normal map 

    the RED channel of the result contain the X value of the normals, the GREEN channel contain the Y values,
    and the BLUE channel contains alpha - a max value of 255 for forground object, and 0 for background
    (where there are no normals)
    """
    result = np.array(result)
    matcap = np.array(matcap)
    matcap_size = len(matcap)

    # normalize
    result = np.concatenate(((result[:,:,:2]/255)*2 - 1,(result[:,:,2:]/255)*2 - 1),axis=2)

    # apply matcap
    foreground_mask = (result[:,:,2] > 0)[:,:,np.newaxis]
    result_x_normals = np.round((result[:,:,0]*0.5 + 0.5)*(matcap_size - 1)).flatten().astype(int)
    result_y_normals = np.round((result[:,:,1]*0.5 + 0.5)*(matcap_size - 1)).flatten().astype(int)
    final_array = matcap[result_x_normals,result_y_normals,:].reshape(result.shape)*foreground_mask # use normals as indexes into the matcap
    return final_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser("this script applies a matcap to a normal map")
    parser.add_argument("target",type=Path,help='the target normal map')
    parser.add_argument("matcap",type=Path,help='the matcap to apply')
    args = parser.parse_args()
    img = Image.open(args.target)
    matcap = Image.open(args.matcap)

    final = apply_matcap_unnormalized(img,matcap)
    Image.fromarray(final).show()
    
    
