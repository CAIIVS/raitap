## MVP Definition

In order to establish a basic, usable tool we design a couple of simple use cases for model assessment (in the transparency dimension).

### Core

- Set up `hydra` configuration structure
- Integrate methods from following frameworks
  - [SHAP](https://shap.readthedocs.io/en/latest/)
  - [Captum](https://captum.ai/api/index.html)
- Integrate performance metrics (classification, object detection, etc.)
- Automate tracking assessment experiments with MLFlow
- Use models from `torchvision`, e.g.
  - classification: `resnet50`
  - classification: `vit_b_32`
  - object detection: `fasterrcnn_resnet50_fpn_v2`
  - integrate ONNX model loading/saving, test methods in ONNX runtime
- Automated reporting solution (suitable for certification auditors)
- Set up package documentation (with mkdocs; see example [c3li](https://github.com/CAIIVS/chuchichaestli))
- Publish on PyPI (automate e.g. [c3li workflow](https://github.com/CAIIVS/chuchichaestli/tree/main/.github/workflows))


### Extensions
- Integrate more methods from following frameworks
  - [OmniXAI](https://opensource.salesforce.com/OmniXAI/latest/index.html)
  - [eth-sri/eran](https://github.com/eth-sri/eran)
  - [SeldonIO/alibli](https://docs.seldon.ai/alibi-explain)
- More use cases (models, datasets)
- Intrinsic explanations for (vision) transformers
- Unit tests (CI/CD)
- simple GUI for workflow guidance


## Datasets

### Dermatology Dataset
Based on a past data challenge this benchmark dataset holds dermatoscopic images from different populations acquired and stored by different modalities [https://doi.org/10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161). It is released as training dataset for academic ML purposes and publicly available through the ISIC archive.

Download: https://challenge.isic-archive.com/landing/2018/


### Malaria Dataset

The data sets behind the work on PlasmoCount (first described in the preprint of our paper on MedRxiv - [https://doi.org/10.1101/2021.01.26.21250284](https://doi.org/10.1101/2021.01.26.21250284). More information about the data sets or model can be gained from contacting the Baum laboratory.

Download: https://data.mendeley.com/datasets/j55fyhtxn4/2


### Udacity Self Driving Car Dataset
The dataset contains 97,942 labels across 11 classes and 15,000 images. There are 1,720 null examples (images with no labels).

All images are 1920x1200 (download size ~3.1 GB). We have also provided a version downsampled to 512x512 (download size ~580 MB) that is suitable for most common machine learning models (including YOLO v3, Mask R-CNN, SSD, and mobilenet).

Annotations have been hand-checked for accuracy by Roboflow.

Download: https://public.roboflow.com/object-detection/self-driving-car
