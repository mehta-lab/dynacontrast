# DynaContrast

Dynacontrast is a pipeline for single cell imaging phenotyping. Dynacontrast uses self-supervised learning to learn a general representation of the cells based on the similarity of cells without labels. The learned representation can then be used to identify cell phenotypes even without prior knowledge of what phenotypes to look for. 

## Dynacontrast pipeline

![pipeline_fig](pipeline.png)

Example UMAP of representations of 4 cell types. Cells that look similarly are grouped together.  

![umap_fig](UMAP_4_cell_types.png)

## Setup the environment
### Using conda in Linux
Clone the repository:

```
git clone https://github.com/mehta-lab/dynacontrast.git
```

Go the repository directory, switch he branch if running branch other than `master`: 

```
cd dynacontrast
git checkout <branch you'd like to run>
```

Create a `conda` environment from the yaml file and activate it:

```
conda env create --file=conda_environment.yml
conda activate dynacontrast
```

Add the repository to python path:

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
### Using Docker on IBM Power PC
Docker image for running dynacontrast has been built on `hulk` (`dynacontrast:<version number>`). Check the newest version by doing:
```
docker image ls dynacontrast
```

To start an interactive container from the image:
```
docker run -it -m 128G --name [name of your container] -v [folder on the host]:[folder inside docker] -p [port on the host]:[port inside docker] --env ACTIVATE=tensorflow --env LICENCE=yes dynacontrast:<version number> bash
```

### Setting up conda env on HPC (A30 & A100 GPUs)
Create a `conda` environment from the yaml file and activate it:
```
conda env create --file=conda_environment_hpc.yml
conda activate dynacontrast
```
Install required packages using pip
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboard==1.15.0
pip install protobuf==3.20.1  #downgrade protobuf is required to work
pip install umap-learn
```

## Usage
You can find the example config file in `examples` directory.

# Proprocessing
To preprocess the images to extract single cell patches from nucleus instance segmentation maps:

	python run_preprocess.py --config <path-to-your-config-yaml>

This will output single cell patches as a single zarr file “cell_patches_all.zarr” if no split is specified in the config. If split is specified, the patches will be split into training and validation sets “cell_patches_train.zarr” and “cell_patches_val.zarr” for model training. 
See `preprocess` section in "examples/config_example.yml" for complete parameters for preprocessing.

Preprocess CLI also outputs single cell patch metadata in .csv format (“patch_meta_all.csv”). The metadata contains patch information such as cell ID, cell position, cell size, position, slice, time. Additional metadata of the experiment (e.g. condition for each well in a multi-well plate) can be provided in the input folder as “metadata.csv” on imaging position level with each position having a unique position ID. This additional metadata will be merged with dynacontrast’s patch metadata. 

(Optional) The preprocessing module processes each 2D image separately, so for z-stack or time-lapse where a cell can span multiple z-slices or frames, multiple patches can be generated from a single cell. 
To link these patches to the same cell, you can add `track_dim` parameter with value 'time' or 'slice' in the config to link patches in time or z dimension.

# Data pooling (optional)
Often times training data needs to combine multiple datasets to improve the model performance. Data pooling CLI provides a simple interface to combine “cell_patches_all.zarr” files and "patch_meta_all.csv" from different experiments into one by supplying in the config the directories of datasets to pool and the destination directory.  

    python run_data_pooling.py --config <path-to-your-config-yaml>

Split parameters can be added to the config in the same way as preprocess config to split the pooled dataset into training and validation sets.   

# Training
Train a model to learn single cell representation of the patches:

	python run_training.py --config <path-to-your-config-yaml>

This will save the model weights as pytorch checkpoint file “model.pt” in “weights_dir” under “model_name” specified in training config. The module also writes tensorboard log file in the same directory that can be visualized using tensorboard.

# Encoding (inference)
Encode image patches into the vectors using the trained model:

	python run_encoding.py --config <path-to-your-config-yaml>

This will output the patch embeddings as `<split>_embeddings.npy`, depending on which split is specified in the config.  

For example, to run encoding using the example config:

    python run_encoding.py --config examples/config_example.yml

•	The module requires “cell_patches_<split>_.zarr” in the “raw_dirs” and model weights in “weights_dirs”.
•	The module outputs the patch embeddings as `<split>_embeddings.npy` under raw_dir/model_name
•	Encoding CLI supports batch processing. Multiple inputs & models can be specified and encoding will run on each input & model combination. If “raw_dirs” has m inputs and “weights_dirs” has n inputs, then total m x n embedding files will be generated.  


# Dimensionality reduction
To reduce the dimension of embeddings for visualization or downstream analysis:

    python run_dim_reduction.py --config <path-to-your-config-yaml>

•	The module requires patch embeddings  `<split>_embeddings.npy` under raw_dir/model_name to run.
•	The module outputs reduced embeddings as `umap_<parameters>.npy` under raw_dir/model_name. 
•	The CLI supports batch processing.  Multiple inputs & models can be specified, and dimensionality reduction will run on each input & model combination. 


