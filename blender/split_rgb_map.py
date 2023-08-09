from PIL import Image
import matplotlib.pyplot as plt
import sys

def display_rgb_maps(image):
    # Convert the image to RGB mode (in case it's not already)
    image = image.convert("RGB")

    # Split the image into its red, green, and blue channels
    red, green, blue = image.split()

    # Create a figure to display the images side by side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Display the red channel
    axes[0].imshow(red, cmap='Reds')
    axes[0].set_title('Red Channel')
    axes[0].axis('off')

    # Display the green channel
    axes[1].imshow(green, cmap='Greens')
    axes[1].set_title('Green Channel')
    axes[1].axis('off')

    # Display the blue channel
    axes[2].imshow(blue, cmap='Blues')
    axes[2].set_title('Blue Channel')
    axes[2].axis('off')

    # Show the plots
    plt.show()

image_path = sys.argv[1]
image = Image.open(image_path)

# Display the RGB maps of the image
display_rgb_maps(image)
