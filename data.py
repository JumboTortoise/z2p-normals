from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# get_image_from_point_cloud performs preprocessing that is specific to our dataset
from convert_point_cloud import get_image_from_point_cloud

RANGES = [[0, 540], [100, 960]]
TARGET = [500, -1]


def resize(img,target = None):
    if target is None:
        target = TARGET
    if target[1] == -1:
        r = img.shape[0] / img.shape[1] 
        img = cv2.resize(img, (target[0], int(r * target[0])))
    else:
        img = cv2.resize(img, (target[0], target[1]))

    return img


class BinMapper:
    """
    Map binned data to global index
    """

    def __init__(self, counts: [int]):
        self.total_elements = sum(counts)
        self.inverse_map = []
        for i, c in enumerate(counts):
            for j in range(c):
                self.inverse_map.append((i, j))

    def __len__(self):
        return len(self.inverse_map)

    def __getitem__(self, item):
        return self.inverse_map[item]


class Shape:

    @staticmethod
    def find_npys(view_folder: Path):
        """
        Find how many examples are for a certain view
        :param view_folder: path to view folder
        :return: int number of examples, 0 if 'render.png' not found
        """
        if not (view_folder / 'normals.png').exists():
            return 0

        counter = 0
        for npy in view_folder.iterdir():
            spt = npy.name.split('.')
            #all_numbers = all(['0' <= c <= '9' for c in spt[0]])
            if len(spt) == 2 and spt[1] == 'npy' and not spt[0].endswith('cache'):
                # yield npy
                counter += 1
        return counter

    def __init__(self, folder):
        self.folder = Path(folder)
        self.views = []
        self.counts = []

        # find views
        for f in self.folder.iterdir():
            if f.name.startswith('rotation_'):
                t_count = Shape.find_npys(f)
                # if there are examples for this view add the it to shapes valid views
                if t_count > 0:
                    self.views.append(f)
                    self.counts.append(t_count)

        # Map the examples per view to a global index
        self.inverter = BinMapper(self.counts)

    def __len__(self):
        # This returns the total number of examples for all views
        return len(self.inverter)

    def __getitem__(self, item):
        # map between global index to a specific example in a specific view
        view_index, item_index = self.inverter[item]
        return self.views[view_index] / f'normals.png', self.views[view_index] / f'cloud_{item_index}.npy'


class GenericDataset(Dataset):
    def __init__(self, folder: Path,
                 splat_size=3, cache=False):
        self.folder = Path(folder)
        self.splat_size = splat_size
        self.shape_paths = list(self.folder.iterdir())
        self.shapes = []
        self.cache = cache
        # read all shapes with positive amount of views
        for s in self.shape_paths:
            t_shape = Shape(s)
            if len(t_shape) > 0:
                self.shapes.append(t_shape)

        self.inverter = BinMapper([len(x) for x in self.shapes])
        

    def __len__(self):
        # This returns the total number of examples for all shapes
        return len(self.inverter)

    def __getitem__(self, item):
        # map between global index to a specific example in a specific shape
        shape_index, item_index = self.inverter[item]
        img_path, z_buffer_path = self.shapes[shape_index][item_index]
        rtn = load_files(img_path, z_buffer_path, splat_size=self.splat_size, cache=self.cache)
        if rtn is None:
            return self.__getitem__(0)

        img, zbuffer = rtn

        return img, zbuffer


def scatter(u_pix, v_pix, distances, res, radius=5, dr=(0, 0), const=9, scale_const=0.7):
    distances -= const
    img = np.zeros(res)
    for (u, v, d) in zip(u_pix, v_pix, distances):
        v, u = round((v - dr[0])), round(u - dr[1])
        f = np.exp(-d / scale_const)
        if radius == 0:
            img[v, u] = max(img[v, u], f)
        else:
            for t1 in range(-radius, radius):
                for t2 in range(-radius, radius):
                    ty, tx = v - t1, u - t2
                    ty, tx = max(0, ty), max(0, tx)
                    ty, tx = min(res[0] - 1, ty), min(res[1] - 1, tx)
                    img[ty, tx] = max(img[ty, tx], f)

    return img


def parse_pts(ar, radius=3, dr=(0, 0), const=9):
    """
    produce a z_buffer from a numpy array of points in relative coordinates
    :param ar: points array NX3 (** with last point representing resolution)
    :param radius: point radius on screen
    :return: np.array NXM with exponential z values for pixel, 0 is inf/background
    """
    res = ar[-1, :-1]
    res = res.astype(int)
    ar = ar[:-1]
    x_target, y_target = int(res[0]), int(res[1])
    return scatter(ar[:, 0], ar[:, 1], ar[:, 2], (x_target, y_target), radius=radius, dr=dr, const=const)


def load_files(png_path, npy_path, splat_size=5, cache=True, dr=(0, 0)):
    """
    load a png target and render a z_buffer from an npy_path
    :param png_path: path to the png target image
    :param npy_path: path to the npy containg the points
    :param cache: if True used caches Z-buffers else False
    :return: (torch.Tensor, torch.Tensor) target_image (3XNXM), z_buffer (1XNXM)
    """
    img = None
    im_height,im_width = None,None
    if png_path is not None:
        img = cv2.imread(str(png_path), cv2.IMREAD_COLOR)

        if img is None:
            return None
        

        # reshape and resize
        im_height,im_width = img.shape[0],img.shape[1]
        img = img[RANGES[0][0]: RANGES[0][1], RANGES[1][0]:RANGES[1][1], :]
        img = resize(img)
        
        

        # move to pytorch tensor and normalize
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1) / 255 # move to pytorch image format and normalize to [0,1]
        #img = img * 2 - 1 # normalize to [-1,1]

        # fix noise that was introduced by blender during dataset creation,
        # alpha (blue channel) should be -1 for background and 1 for fourground
        #img[-1,:,:][img[-1,:,:] > 0.5] = 1
        #img[-1,:,:][img[-1,:,:] <= 0.5] = -1


    if cache:
        cache_path = npy_path.parent / npy_path.name.replace('.npy', '_cache.npy')
        if cache_path.exists():
            z_buffer = np.load(cache_path)
        else:
            if img is None:
                return None
            ptsD = np.load(str(npy_path))
            ptsD = get_image_from_point_cloud(ptsD,50,im_height,im_width) # our preprocessing
            z_buffer = parse_pts(ptsD, radius=splat_size)
            np.save(cache_path, z_buffer)
    else:
        if img is None:
                return None
        ptsD = np.load(str(npy_path))
        ptsD = get_image_from_point_cloud(ptsD,50,im_height,im_width) # our preprocessing
        z_buffer = parse_pts(ptsD, radius=splat_size, dr=dr)

    z_buffer = z_buffer[RANGES[0][0]: RANGES[0][1], RANGES[1][0]:RANGES[1][1]]
    z_buffer = resize(z_buffer)

    z_buffer = torch.from_numpy(z_buffer)
    z_buffer = z_buffer.unsqueeze(0)

    if png_path is not None:
        #print("load files:",img.shape,z_buffer.shape)
        return img, z_buffer
    else:
        return z_buffer
