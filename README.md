# Neural Volume Super Resolution
<!--- #### Official PyTorch implementation
### Project page (Coming soon) | [Paper](https://arxiv.org/abs/2212.04666) -->


## Requitements
Begin by setting up the dependencies. You can create a conda environment using `conda env create -f environment.yml`. Then update the root path in the [local configuration file](config/local_config.yml.example), and remove its `.example` suffix. Install [torchsearchsorted](https://github.com/aliutkus/torchsearchsorted) by following instructions from their `README`.
## Super-resolve volumetric scene(s)
Our framework includes three learned components: A decoder model and a feature-plane super-resolution model shared between all 3D scenes, and an individual set of feature planes per 3D scene. You can experiment with our code in different levels, by following the directions starting from any of the 3 possible stages below (directions marked with * should only be perfomed if starting from the stage they appear in):
### Train everything from scratch
1. Download our [training scenes dataset](https://drive.google.com/file/d/10F2SPY-laYzdNzdNrxa_Yd4KA3qLbK8z/view?usp=sharing).
1. Download the desired (synthetic) test scene from the [NeRF dataset](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) and put all scenes in a dataset folder.
1. Update the [configuration file](config/TrainModels.yml). Add the desired test scene name(s) to the [training list](config/TrainModels.yml#L50). Update the scene name(s) in the [evaluation list](config/TrainModels.yml#L54) and update the paths to the [scenes dataset folder](config/TrainModels.yml#L20) and to storing the [new models](config/TrainModels.yml#L4) in the configuration file.
1. Run `python train_nerf.py --config config/TrainModels.yml`


### Super-resolve a new test scene
Use pre-trained decoder and plane super-resolution models while learning feature planes corresponding to a new 3D scene.
1. Download our [pre-trained models file](https://drive.google.com/file/d/1zdod2hVQO8H3WzGfzMuvKEkbenGUbrkA/view?usp=sharing) and unzip it.
1. *Download our [training scenes dataset](https://drive.google.com/file/d/10F2SPY-laYzdNzdNrxa_Yd4KA3qLbK8z/view?usp=sharing).
1. *Download the desired (synthetic) test scene from the [NeRF dataset](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) and put all scenes in a dataset folder.
1. Learn the feature planes representation for a new test scene:
    1. Update the [configuration file](config/Feature_Planes_Only.yml). Add the desired test scene name(s) to the [training list](config/Feature_Planes_Only.yml#L50). Then update the scene name(s) in the [evaluation list](config/Feature_Planes_Only.yml#L54), as well as the paths to the [scenes dataset folder](config/Feature_Planes_Only.yml#L22), [pre-trained models folder](config/Feature_Planes_Only.yml#L61) and to storing the [new scene feature planes](config/Feature_Planes_Only.yml#L4) in the configuration file.
    1. Run `python train_nerf.py --config config/Feature_Planes_Only.yml`
1. Jointly refine all three modules:
    1. Update the desired scene name ([training](config/RefineOnTestScene.yml#L53) and [evaluation](config/RefineOnTestScene.yml#L57)), as well as the paths to the [scenes dataset folder](config/RefineOnTestScene.yml#L20), pre-trained models folder ([decoder](config/RefineOnTestScene.yml#L65) and [SR](config/RefineOnTestScene.yml#L180)), [learned scene feature planes](config/RefineOnTestScene.yml#L67) (from the previous step) and to storing the [refined models](config/RefineOnTestScene.yml#L4) in the configuration file.
    1. Run `python train_nerf.py --config config/RefineOnTestScene.yml`
### Evaluate a pre-learned test scene
Use pre-trained decodeer and SR models, coupled with the learned feature-plane representation:
1. *Download one of our [pre-trained models](https://drive.google.com/drive/folders/1ZWRazAZ21nLUsdsfYOmnyBtkYFYV-SVm?usp=sharing) and unzip it, then download the corresponding ([synthetic](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) or [real world](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7)) scene from the NeRF dataset.
1. Run: 
    ```
    python train_nerf.py --load-checkpoint <path to pre-trained models folder> --eval video --results_path <path to save output images and video>
    ```




Optionally, to resume training in any of the first two stages, use the `--load-checkpoint` argument followed by the path to the saved model folder, and omit the `--config` argument.

<!--- 
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
-->
## Contributing / Issues?

Feel free to raise GitHub issues if you find anything concerning. Pull requests adding additional features are welcome too.


## LICENSE

This code is available under the [MIT License](https://opensource.org/licenses/MIT). The code was forked from the [nerf-pytorch](https://github.com/krrish94/nerf-pytorch) repository.
 <!-- For more details see: [LICENSE](LICENSE) and [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS). -->
