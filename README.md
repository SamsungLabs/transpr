# TRANSPR: Transparency Ray-Accumulating Neural 3D Scene Point Renderer
### <img align=center src=./docs/images/project.png width='32'/> [Project page](https://saic-violet.github.io/transpr) &ensp; <img align=center src=./docs/images/paper.png width='24'/> [Paper](https://arxiv.org/pdf/2009.02819.pdf) &ensp; <img align=center src=./docs/images/video.png width='24'> [Video](https://www.youtube.com/watch?v=VJ_JFPCiafc) &ensp; <img align=center src=./docs/images/video.png width='24'> [Talk @ 3DV](https://youtu.be/aiMm4n1-izg) 

[TRANSPR: Transparency Ray-Accumulating Neural 3D Scene Point Renderer](https://arxiv.org/pdf/2009.02819.pdf)<br>
[Maria Kolos*](https://github.com/mvkolos)<sup>1</sup> &nbsp;
[Artem Sevastopolsky*](https://seva100.github.io)<sup>1,2</sup> &nbsp;
[Victor Lempitsky](http://sites.skoltech.ru/compvision/members/vilem/)<sup>1,2</sup> <br>
<sup>1</sup>Samsung AI Center &nbsp; <sup>2</sup>Skolkovo Institute of Science and Technology <br><br>
\* *indicates equal contribution*

<br>

<img src=docs/images/teaser.png width=1200>

## About

This is a PyTorch implementation of TRANSPR, a new method for realtime photo-realistic rendering of 3D scenes with semi-transparent parts and complex geometry. This method is an extension of [Neural Point-Based Graphics](https://saic-violet.github.io/npbg) (NPBG) which uses a raw point cloud as the geometric representation of a scene, and augments each point with a learnable neural descriptor that encodes local geometry and appearance. In our method, we expand the descriptor with a learned transparency value, use ray accumulation to account all points perceived by the camera. The points along the rays are fused into an opacity-aware 2D representation processed by the rendering network. Several scenes can be rendered in conjunction, opacity of objects can be edited, and the non-transparent objects can be combined with the introduced transparency. The repository extends NPBG with the additional features. 

<img src=docs/images/pipeline.png width=1200>

## Environment

Run this command to install python environment using [conda](https://docs.conda.io/en/latest/miniconda.html) (you might need to manually change the 'cudatoolkit' version according to CUDA installed on the machine):
```bash
source scripts/install_deps.sh
```

## Quick start

Download the datasets and pretrained weights from [here](https://drive.google.com/file/d/1TmY0me9uZUpOaxRdeOGplcSFIo82e2_Q/view?usp=sharing) and unpack in the sources root directory.

We suppose that you have at least one GeForce GTX 1080 Ti for fitting and inference. This code does not use OpenGL rendering, so can be run in headless mode.

You can run inference using the pretrained weights and configs in [configs/infer](https://github.com/mvkolos/npbg/tree/transpr/configs/infer):
```bash
python eval.py --config configs/infer/<INFERENCE_CONFIG> --save_dir <SAVE_DIR>
```

Or train the models yourself using configs in [configs/train](https://github.com/mvkolos/npbg/tree/transpr/configs/train):

```bash
python train.py --config configs/train/<TRAIN_CONFIG>
```
### Configs structure
Please refer to [configs](https://github.com/mvkolos/npbg/tree/transpr/configs) folder for particular dataset configs and detailed explanations for [train](https://github.com/mvkolos/npbg/blob/transpr/configs/train/train_explained.yaml) and [inference](https://github.com/mvkolos/npbg/blob/transpr/configs/infer/inference_explained.yaml) (viewer.py is not yet integrated into the current pipeline, so eval.py is the only way to render results).

## Data
### Scene types and training strategies
Scenes can either be synthetic (rendered in e.g. Blender) or real ones obtained via photogrammetry. 
The pipeline allows to combine all kinds of scenes into compositions. 
Assuming the final scene type (composition or single one) the train strategy is the following:
1. If you want to train scenes for composition turn on [random_overlay](https://github.com/mvkolos/npbg/blob/transpr/configs/train/train_explained.yaml#L77) parameter in [train_dataset_args](https://github.com/mvkolos/npbg/blob/transpr/configs/train/train_explained.yaml#L63);
2. If case you plan manipulating alpha turn on [alpha_jitter](https://github.com/mvkolos/npbg/blob/transpr/configs/train/train_explained.yaml#L72) parameter in [train_dataset_args](https://github.com/mvkolos/npbg/blob/transpr/configs/train/train_explained.yaml#L63). Make sure [density_scale](https://github.com/mvkolos/npbg/blob/transpr/configs/train/train_explained.yaml#L33) is present in [texture_args](https://github.com/mvkolos/npbg/blob/transpr/configs/train/train_explained.yaml#L28). We used value of 1 for initialization;
3. If your images are segmented or masked and contain 4 channels you can train pipeline with [random_background](https://github.com/mvkolos/npbg/blob/transpr/configs/train/train_explained.yaml#L9) on to enable learning of output alpha (note that you can still use 3-channel loss, so [l1_weight](https://github.com/mvkolos/npbg/blob/transpr/configs/train/train_explained.yaml#L95) is 0). Make sure [num_output_channels](https://github.com/mvkolos/npbg/blob/transpr/configs/train/train_explained.yaml#L24) is set to 4 in this case.

### Creating your own scenes
### Real
Please refer to the instructions in the [original](https://github.com/alievk/npbg#fit-your-scene) pipeline to build a point cloud. In order to leverage alpha learning and  [random_background](https://github.com/mvkolos/npbg/blob/transpr/configs/train/train_explained.yaml#L9) you additionally need to generate masks for the RGB images used for reconstruction. You may have initially segmented images by using any background removing software (e.g. https://www.remove.bg/ru or [Select subject](https://helpx.adobe.com/ru/photoshop/how-to/select-subject-one-click.html) in Photoshop) or in Agisoft Metashape by first, reconstructing a mesh: *Workflow->Build Mesh (from dense point cloud)* then *Tools->Mesh->Generate masks (Replacement)* and finally *File->Export->Export Masks*. You will then need to create 4-channel images with RGB channels multiplied by alpha mask (from 0 to 1.0).

### Synthetic

We used [this tutorial](https://youtu.be/_IDBfQnttiE) to generate dynamic and static smoke. Feel free to experiment with blender or other rendering software but ensure the final images have RGB values multiplied by alpha values (from 0 to 1.0). Don't pay attention if your system image viewer displays twice more transparent image.
