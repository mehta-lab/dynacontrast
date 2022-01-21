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
You can find the template and the example config file in `examples` directory in the repository.

Extract single cell patches from single cell instance segmentation maps, then connect them into trajectories:

	python run_patch.py -m "extract_patches" -c <path-to-your-config-yaml>
	python run_patch.py -m "build_trajectories" -c <path-to-your-config-yaml>

Assemble extracted patches into a single zarr array:

	python run_VAE.py -m "assemble" -c <path-to-your-config-yaml>

Train a model to learn single cell representation of the patches:

	python run_training.py -c <path-to-your-config-yaml>

Encode image patches into the vectors using the model:

	python run_VAE.py -m "process" -c <path-to-your-config-yaml>

For example, to run encoding using the example config:

    


