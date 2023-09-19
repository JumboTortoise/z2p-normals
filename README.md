# Z2P-Normals: Point Cloud Visualization With Material Captures

This is an adaptation of the work presented in this [paper](https://arxiv.org/abs/2105.14548), for the use case of producing normal maps from point clouds.
### Installation
- Clone this repository
- Create a new python virtual environment. The code was tested on python version 3.11.5
- Install the dependencies from the requirements.txt file
# Running 

## Data
Four different datasets were used in the project, three for training and one for testing.
All datasets are provided as .zip files, and need to be extracted before training. <br> 
Each set should take up about 12GB of disk space. 
If you wish to use the ***cache option*** at train time please space for around 350GB of disk space. <br>
This will save the 2D point cloud z-buffers to disk and allow for faster training. 

All the datasets and pretrained checkpoints are available in [this](https://drive.google.com/drive/u/1/folders/1WOLzSjL7GS3M9YbbMzg8xIsXxfNXVKg-) google drive directory
(Like this repo, it is private, contact us if you require permissions).

## Training

The main training script is ``train.py``.
An example training command is:

``python train.py --data /bhdd/Datasets/normals_80k_augmented/ --export-dir ./models/ --batch-size 16 --num-workers 4 --epochs 10 --log-iter 1000 --losses masked_mse intensity masked_pixel_intensity masked_laplacan_weighted_mse --l-weight 1 0.7 1 1.2 --splat-size 1 --mixed-precision --cache --aug-passes 1 --aug-null 0.8 --aug-noise 0.2 --arch``

This command will train a model on the 80K examples augmented dataset(In this example, it is located in /bhdd/Datasets/normals_80k_augmented) in mixed precision mode for 10 epochs, with a batch size of 16 and z-buffer caching enabled. The description of all possible options can be made visible by running ``python train.py --help``.

## Inference
Inference is done with the ``inference_pc.py`` script.
Every checkpoint has a corresponding settings.json file, make sure that the correct settings are used.
advanced options are accessible through ``python inference_pc.py --help``.
