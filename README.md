python version 3.7.10

torch version 1.9.0

numpy version 1.21.0

The three folders C10, C100, ImageNet provide corresponding code for CIFAR-10, CIFAR-100 and ImageNet datasets.

Please download the corresponding datasets and save them to the respective folders. 

Alternatively, you can use PyTorch's built-in dataset downloading capabilities.

To save the decomposed or trained network, define the folder in advance and modify the save path accordingly.

To reproduce the experimental results from the paper, first set the parameters pw_subvector_size and the number of centroids k in the YAML file. Then, adjust the channel compression rate in the Tucker_layer file, and set the self-attention percentage in the lowrank_compress file.
