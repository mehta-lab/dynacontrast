# DynaMorph

% Overview %

### Table of contents:

- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [DynaMorph Pipeline](#dynamorph-pipeline)
  - [Label-free Imaging](#label-free-imaging)
  - [Cell Segmentation and Tracking](#cell-segmentation-and-tracking)
  - [Latent Representations of Morphology](#latent-representations-of-morphology)
- [Citing DynaMorph](#citing-dynamorph)

## Requirements

DynaMorph is developed and tested under Python 3.7, packages below are required.

- [imageio](https://imageio.github.io/)
- [Keras](https://keras.io/)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)
- [OpenCV](https://opencv.org/about/)
- [PyTorch](https://pytorch.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [segmentation-models](https://github.com/qubvel/segmentation_models)
  - Dynamorph requires segmentation-models v0.2.1
- [TensorFlow](https://www.tensorflow.org/)
- [tifffile](https://pypi.org/project/tifffile/)

## Getting Started

% Overview of the tool %

## DynaMorph Pipeline

DynaMorph starts with raw image files from cell imaging experiments and sequentially applies a set of segmentation and encoding tools. Below we briefly introduced the main processing steps.

### Label-free Imaging
(for pipeline, can use data acquired from any microscopy source -- file format as .tif)
(in dynamorph paper, we use phase and retardance)
(collect polarization-resolved label-free images using method in <reference to virtual staining paper> )


### Cell Segmentation and Tracking

Scripts under `NNsegmentation` folder contain codes for U-Net based cell segmentation model. `NNsegmentation/run.py` provides an example on the model training procedure. Usage of the segmentation model in the whole pipeline can be found in `pipeline/segmentation.py`.

Instance segmentation in this work is based on clustering, related methods can be found in `SingleCellPatch/extract_patches.py`. Following cell tracking methods can be found in `SingleCellPatch/generate_trajectories.py`.

To generate segmentation and tracking from scratch, follow steps below:

1. (optional) prepare training images and labels

2. (optional) train a classifier, see scripts in `NNsegmentation/run.py`

3. prepare inputs as 4-D numpy arrays of shape (N<sub>time frames<sub>, height, width, N<sub>channels<sub>), see method `pipeline.preprocess.write_raw_to_npy` for an example

4. apply trained model for semantic segmentation, see method `pipeline.segmentation.segmentation` or `run_segmentation.py` (uncomment [line "segmentation"](https://github.com/czbiohub/dynamorph/blob/8965b5d7b21895d95d548cc3ef6c1a397cee8255/run_segmentation.py#L71))

5. use predicted class probabilities for instance segmentation, see method `pipeline.segmentation.instance_segmentation` or `run_segmentation.py` (uncomment [line "instance_segmentation"](https://github.com/czbiohub/dynamorph/blob/8965b5d7b21895d95d548cc3ef6c1a397cee8255/run_segmentation.py#L72))

6. connect static cell frames to trajectories, see method `pipeline.patch_VAE.build_trajectories` or `run_patch.py` (uncomment [line "build_trajectories"](https://github.com/czbiohub/dynamorph/blob/8965b5d7b21895d95d548cc3ef6c1a397cee8255/run_patch.py#L45))

### Latent Representations of Morphology
DynaMorph uses VQ-VAE to encode and reconstruct cell image patches, from which latent vectors are used as morphology descriptor. Codes for building and training VAE models are stored in `HiddenStateExtractor/vq_vae.py`.

To extract single cell patches and employ morphology encoding, follow steps below:

7. extract cell patches based on instance segmentation, see method `pipeline.patch_VAE.extract_patches` or `run_patch.py` (uncomment [line "extract_patches"](https://github.com/czbiohub/dynamorph/blob/8965b5d7b21895d95d548cc3ef6c1a397cee8255/run_patch.py#L44))  

8. (optional) train a VAE for cell patch reconstruction, see the [main block](https://github.com/czbiohub/dynamorph/blob/8965b5d7b21895d95d548cc3ef6c1a397cee8255/HiddenStateExtractor/vq_vae.py#L1041) of `HiddenStateExtractor/vq_vae.py` for reference.

9. assemble cell patches generate from step 7 to model-compatible datasets, see method `pipeline.patch_VAE.assemble_VAE` or `run_VAE.py` (uncomment [line "assemble_VAE"](https://github.com/czbiohub/dynamorph/blob/8965b5d7b21895d95d548cc3ef6c1a397cee8255/run_VAE.py#L41)) 

10. apply trained VAE models on cell patches, see method `pipeline.patch_VAE.process_VAE` or `run_VAE.py` (uncomment [line "process_VAE"](https://github.com/czbiohub/dynamorph/blob/8965b5d7b21895d95d548cc3ef6c1a397cee8255/run_VAE.py#L42))

## Citing DynaMorph

To cite DynaMorph, please use the bibtex entry below:

```
@article{wu2020dynamorph,
  title={DynaMorph: learning morphodynamic states of human cells with live imaging and sc-RNAseq},
  author={Wu, Zhenqin and Chhun, Bryant B and Schmunk, Galina and Kim, Chang and Yeh, Li-Hao and Nowakowski, Tomasz J and Zou, James and Mehta, Shalin B},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
```
