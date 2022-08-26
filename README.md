# the Artificial Artist


## Set-up

#### Dependencies

Let's install the dependencies in a fresh conda environment:

```shell
conda create --name artist python
conda activate artist 

conda install pytorch torchvision -c pytorch
conda install diffusers transformers scipy ftfy -c conda-forge
```

If you ever want to trash that environment, you can run:

```shell
conda deactivate
conda env remove -n artist
```

#### Weights

We will download the weights from Hugginface.

You will need an account on [huggingface](https://huggingface.co/CompVis) **and** [accepting the terms and conditions for the model](https://huggingface.co/CompVis/stable-diffusion-v1-4) you are using to download the weights.

Then, run the following line in a shell and generate a token to be able to download the weights:

```shell
huggingface-cli login
```

## Usage

To use first activate our conda environment:

```shell
conda activate artist
```

#### diffusionSimple

The `diffusionSimple` folder includes `StableDiffusionPipelineSimple`, a fork of hugginface's `StableDiffusionPipeline` slightly simplified and better documented.
This pipeline was written to be used as a basis for the other ones.

#### diffusionPar

The `diffusionPar` folder includes `StableDiffusionPipelinePar`, a fork of hugginface's `StableDiffusionPipeline` designed to use several GPU at the same time.

#### diffusionLite

The `diffusionLite` folder includes `StableDiffusionPipelineLite`, a fork of hugginface's `StableDiffusionPipeline` designed to reduce memory use.

To do so, it:
- split the pipe into individual functions to garbage collect intermediate results as soon as we leave a local function
- move parts of the model to GPU as they are needed and retrieve them afterward
- do more operations in place
- compute the attention piece-wise to keep things manageable

This was finetuned for my personal laptop and might not be lean enough for yours.

TODO: we could remove allocation from the attention block and keep operations in place to reduce the number of splits needed.

## Credits

This code is based on [Hugginface's diffuser librarie](https://github.com/huggingface/diffusers).

Stable diffusion's original code can be found [here](https://github.com/CompVis/stable-diffusion).
