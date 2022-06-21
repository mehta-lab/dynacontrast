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

## Usage
You can find the example config file in `examples` directory.

# Proprocessing
To preprocess the images to extract single cell patches from nucleus instance segmentation maps:

	python run_preprocess.py --config <path-to-your-config-yaml>

This will output single cell patches as a single zarr file “cell_patches_all.zarr” if no split is specified in the config. If split is specified, the patches will be split into training and validation sets “cell_patches_train.zarr” and “cell_patches_val.zarr” for model training.

# Training
Train a model to learn single cell representation of the patches:

	python run_training.py --config <path-to-your-config-yaml>

This will save the model weights as pytorch checkpoint file “model.pt” in “weights_dir” under “model_name” specified in training config.

# Encoding
Encode image patches into the vectors using the trained model:

	python run_encoding.py --config <path-to-your-config-yaml>

This will output the patch embeddings as `<split>_embeddings.npy`, depending on which split is specified in the config.  

For example, to run encoding using the example config:

    python run_encoding.py --config examples/config_example.yml 

    


