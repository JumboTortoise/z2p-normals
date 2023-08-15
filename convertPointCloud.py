import numpy as np
from PIL import Image
import sys
import time
import matcap


def render_point_cloud(point_cloud, focal_length, image_width, image_height):
    # Define the projection matrix based on camera orientation
    P = np.array([[0, 0, focal_length],
                  [0, focal_length, 0],
                  [1, 0, 0]])

    # Initialize an empty image with white color
    image = np.full((image_height, image_width), 255, dtype=np.uint8)

    # Apply projection to each point in the point cloud
    for point in point_cloud:
        # Perform projection
        projected_point = [1,focal_length,focal_length]*point 
        x, y, w = projected_point

        # Normalize the projected coordinates
        w = 1*w/x + image_width / 2
        y = 1*y/x + image_height / 2

        # Check if the point is within the image boundaries
        if 0 <= w < image_width and 0 <= y < image_height:
            # Set the pixel at (x, y) to black
            image[int(y), int(w)] = 0
        else:
            z=0

    return Image.fromarray(image, 'L')  # 'L' mode for grayscale image


if __name__=="__main__":

    image = Image.open(r"C:\DeepLrProject\normals\gun\rotation_43\normals.png")

    # Convert the image to a NumPy array
    image_array = np.array(image)

    point_cloud = np.load(r"C:\DeepLrProject\normals\gun\rotation_43\cloud_1.npy")

    focal_length = 50  # Adjust according to your needs
    image_width = 960
    image_height = 540

    rendered_image = render_point_cloud(point_cloud, focal_length, image_width, image_height)
    rendered_image.show()

    image_pil = Image.fromarray(image_array)

    # Now 'image_pil' is a PIL image
    image_pil.show()  # Display the PIL image