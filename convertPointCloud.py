import os
import numpy as np
from PIL import Image
import sys
import time
import matcap
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def render_point_cloud(point_cloud, focal_length, image_width, image_height):
    # Define the projection matrix based on camera orientation
    P = np.array([[0, 0, focal_length],
                  [0, focal_length, 0],
                  [1, 0, 0]])

    # Initialize an empty image with white color
    image = np.full((image_height, image_width), 255, dtype=np.uint8)

    # Apply projection to each point in the point cloud
    for point in point_cloud:

        point1 = point

        # point1 = point + [0,0,2] 

        # Perform projection
        projected_point = [1,focal_length,focal_length]*point1 
        x, y, z = projected_point

        # Normalize the projected coordinates
        z = 20*z/x + image_height / 2
        y = 20*y/x + image_width / 2

        z = image_height - z

        y = image_width - y

        # Check if the point is within the image boundaries
        if 0 <= y < image_width and 0 <= z < image_height:
            # Set the pixel at (x, y) to black
            image[int(z), int(y)] = 0
        else:
            z=0

    return Image.fromarray(image, 'L')  # 'L' mode for grayscale image

def get_image_from_point_cloud(point_cloud, focal_length, image_height, image_width):
     # Apply projection to each point in the point cloud
    
    x = [1,focal_length,focal_length]

    point_cloud_projected = point_cloud * x

    point_cloud_projected = point_cloud_projected.T

    result_points = 20 * point_cloud_projected[1:] / point_cloud_projected[0]

    rounded_result = np.floor(result_points)

    new_row = point_cloud.T[0] - focal_length * 0.001
    rounded_result = np.vstack((rounded_result, new_row))

    rounded_result[0] += image_width / 2
    rounded_result[1] += image_height / 2

    rounded_result[0] = image_width - rounded_result[0]
    rounded_result[1] = image_height - rounded_result[1]

    result_points_T = rounded_result.T
    resulotion = np.array([image_height,image_width,0]).T

    result = np.vstack((result_points_T,resulotion))

    return result



def fix_one_folder(file_in):

    focal_length = 50
    image_height = 540 
    image_width = 960

    for filename in os.listdir(file_in):
        if filename.endswith('.npy'):
            file_path = os.path.join(file_in, filename)

            data = np.load(file_path)

            point_cloud_image = get_image_from_point_cloud(data, focal_length, image_height, image_width)

            np.save(file_path, point_cloud_image)


def fix_shape(dir_path):

    subdirectories = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(dir_path, subdirectory)
        fix_one_folder(subdirectory_path)

def draw_fixed_shape(path):
    # Create a random 3x5000 array
    array = np.load(path)

    res = array[-1, :-1]
    res = res.astype(int)
    array = array[:-1]

    image_width = res[1]
    image_height = res[0]

    # Extract the first two rows as x and y coordinates
    x_coords = array[:,0].astype(int)
    y_coords = array[:,1].astype(int)

    # Create a grayscale image as a NumPy array
    image = np.ones((image_height, image_width), dtype=np.uint8) * 255

    # Set the specified pixels to black
    for x, y in zip(x_coords, y_coords):
        image[y, x] = 0  # Set pixel to black (0)

    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(image)

    # Display the PIL Image
    pil_image.show()


if __name__=="__main__":

    shape_dir = r"C:\Users\פישר\Downloads\data_set\chair"
    point_cloud_path = r"C:\Users\פישר\Downloads\data_set\chair\rotation_75\cloud_0.npy"

    # fix_shape(shape_dir)

    draw_fixed_shape(point_cloud_path)

        # Path to the PNG image
    p = r"C:\Users\פישר\Downloads\data_set\chair\rotation_75\normals.png"

    # Open the image using Pillow
    image = Image.open(p)

    # Display the image
    image.show()