# Neural Volume Super Resolution
#### Official PyTorch implementation
### Project page (Coming up soon) | [Paper](https://arxiv.org/abs/2212.04666)


## Requitements
Begin by setting up the dependencies. You can create a conda environment using `conda env create -f environment.yml`. Then update the root path in the [local configuration file](config/local_config.yml.example), and remove its `.example` suffix.
## Super-resolve a volumetric scene
Our framework includes three learned components: A decoder model and a feature-plane super-resolution model shared between all 3D scenes, and an individual set of feature planes per 3D scene. You can experiment with our code in different levels, by following the directions starting from any of the 3 possible stages below (directions marked with * should only be perfomed if starting from the stage they appear in):
1. ### Train everything from scratch
    1. Download our [training scenes dataset](https://drive.google.com/file/d/10F2SPY-laYzdNzdNrxa_Yd4KA3qLbK8z/view?usp=sharing).
    1. Download the desired (synthetic) test scene from the [NeRF dataset](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) and put all scenes in a dataset folder.
    1. Update the [configuration file](config/TrainModels.yml). Add the desired test scene name(s) to the [training list](config/TrainModels.yml#L50). Update the scene name(s) in the [evaluation list](config/TrainModels.yml#L54) and update the paths to the [scenes dataset folder](config/TrainModels.yml#L20) and to storing the [new models](config/TrainModels.yml#L4) in the configuration file.
    1. Run
        ```
        train_nerf.py --config config/TrainModels.yml
        ```


1. ### Super-resolve a new test scene
    Use pre-trained decoder and plane super-resolution models while learning feature planes corresponding to a new 3D scene.
    1. *Download our [pre-trained models file]() and unzip it.
    1. *Download our [training scenes dataset](https://drive.google.com/file/d/10F2SPY-laYzdNzdNrxa_Yd4KA3qLbK8z/view?usp=sharing).
    1. *Download the desired (synthetic) test scene from the [NeRF dataset](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) and put all scenes in a dataset folder.
    1. Learn the feature planes representation for a new test scene:
        1. Update the [configuration file](config/Feature_Planes_Only.yml). Add the desired test scene name(s) to the [training list](config/Feature_Planes_Only.yml#L55). Then update the scene name(s) in the [evaluation list](config/Feature_Planes_Only.yml#L59), as well as the paths to the [scenes dataset folder](config/Feature_Planes_Only.yml#L22), [pre-trained models folder](config/Feature_Planes_Only.yml#L61) and to storing the [new scene feature planes](config/Feature_Planes_Only.yml#L4) in the configuration file.
        1. Run
            ```
            train_nerf.py --config config/Feature_Planes_Only.yml 
            ```
    1. Jointly refine all three modules:
        1. Update the desired scene name ([training](config/RefineOnTestScene.yml#L53) and [evaluation](config/RefineOnTestScene.yml#L57)), as well as the paths to the [scenes dataset folder](config/RefineOnTestScene.yml#L20), [pre-trained models folder](config/RefineOnTestScene.yml#L65), [learned scene feature planes](config/RefineOnTestScene.yml#L67) (from the previous step) and to storing the [refined models](config/RefineOnTestScene.yml#L4) in the configuration file.
        1. Run
            ```
            train_nerf.py --config config/RefineOnTestScene.yml
            ```
1. ### Evaluate a pre-learned test scene
    Use pre-trained decodeer and SR models, coupled with the learned feature-plane representation:
    1. *Download one of our [pre-trained models](https://drive.google.com/drive/folders/1rHR5s1JUdtayk7kEcjONdv6MzJd1L-6F?usp=sharing) and unzip it, then download the corresponding ([synthetic](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) or [real world](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7)) scene from the NeRF dataset.
    1. Run:
        ```
        python train_nerf.py --load-checkpoint <path to pre-trained models folder> --eval video --results_path <path to save output images and video> 
        ```


The NeRF code release has an accompanying Colab notebook, that showcases training a feature-limited version of NeRF on a "tiny" scene. It's equivalent PyTorch notebook can be found at the following URL:

https://colab.research.google.com/drive/1rO8xo0TemN67d4mTpakrKrLp03b9bgCX


## How to train your NeRF

To train a "full" NeRF model (i.e., using 3D coordinates as well as ray directions, and the hierarchical sampling procedure), first setup dependencies. In a new `conda` or `virtualenv` environment, run
```
pip install requirements.txt
```

**Importantly**, install [torchsearchsorted](https://github.com/aliutkus/torchsearchsorted) by following instructions from their `README`.

Once everything is setup, to run experiments, first edit `config/default.yml` to specify your own parameters.

The training script can be invoked by running
```
python train_nerf.py --config config/default.yml
```

Optionally, if resuming training from a previous checkpoint, run
```
python train_nerf.py --config config/default.yml --load-checkpoint path/to/checkpoint.ckpt
```


## (Full) NeRF on Google Colab

A Colab notebook for the _full_ NeRF model (albeit on low-resolution data) can be accessed [here](https://colab.research.google.com/drive/1L6QExI2lw5xhJ-MLlIwpbgf7rxW7fcz3).


## A note on reproducibility

All said, this is not an official code release, and is instead a reproduction from the original code (released by the authors [here](https://github.com/bmild/nerf)).

I have currently ensured (to the best of my abilities, but feel free to open issues if you feel something's wrong :) ) that
* Every _individual_ module exactly (numerically) matches that of the TensorFlow implementation. [This Colab notebook](https://colab.research.google.com/drive/1ENrAtZIEhoeNkaXOXkBL7SbWU1VWHBQm) has all the tests, matching op for op (but is very scratchy to look at)!
* Training works as expected for fairly small resolutions (100 x 100).

However, this implementation still **lacks** the following:
* I have not run all the full experiments devised in the paper.
* I've only tested on the `lego` sequence of the synthetic (Blender) datasets.

The organization of code **WILL** change around a lot, because I'm actively experimenting with this.

**Pretrained models**: I am running a few large-scale experiments, and I hope to release models sometime in the end of April.

## Citing
If you find our code or paper useful, please consider citing
```bibtex
@misc{bahat2022neural,
      title={Neural Volume Super-Resolution}, 
      author={Yuval Bahat and Yuxuan Zhang and Hendrik Sommerhoff and Andreas Kolb and Felix Heide},
      year={2022},
      eprint={2212.04666},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Contributing / Issues?

Feel free to raise GitHub issues if you find anything concerning. Pull requests adding additional features are welcome too.


## LICENSE

This code is available under the [MIT License](https://opensource.org/licenses/MIT). The code was forked from the [nerf-pytorch](https://github.com/krrish94/nerf-pytorch) repository.
 <!-- For more details see: [LICENSE](LICENSE) and [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS). -->
