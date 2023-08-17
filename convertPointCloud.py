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


if __name__=="__main__":

    image = Image.open(r"C:\DeepLrProject\normals\chair\rotation_0\normals.png")

    # Convert the image to a NumPy array
    image_array = np.array(image)

    point_cloud = np.load(r"C:\DeepLrProject\normals\chair\rotation_0\cloud_0.npy")

    focal_length = 50  # Adjust according to your needs
    image_width = 960
    image_height = 540

    rendered_image = render_point_cloud(point_cloud, focal_length, image_width, image_height)
    rendered_image.show()

    image_pil = Image.fromarray(image_array)

    # Now 'image_pil' is a PIL image
    image_pil.show()  # Display the PIL image

    # Create a new figure and add a 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract X, Y, Z coordinates from the point cloud
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # Create a 3D scatter plot
    ax.scatter(x, y, z, c='b', marker='o')

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()