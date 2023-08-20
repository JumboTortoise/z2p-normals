import torch
import data
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
def get_data_loader():

    
    train_set = data.GenericDataset(r"/hdd/Datasets/normals", splat_size=3, cache=False)

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    
    return train_set
    

if __name__=="__main__":
    # array = np.load(r"C:\Users\פישר\Downloads\train_set\renders_shade_abs\koala\view_0\0.npy")

    data_loader = get_data_loader()

    r_image, point_cloud_img = data_loader[20]

    print(r_image)

    # Convert tensor to PIL Image
    tensor_to_pil = transforms.ToPILImage()(r_image)

    # Display the PIL Image
    tensor_to_pil.show()

    # Convert tensor to PIL Image
    tensor_to_pil = transforms.ToPILImage()(point_cloud_img)

    # Display the PIL Image
    tensor_to_pil.show()

    
    


    