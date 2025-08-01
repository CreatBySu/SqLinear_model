<div align="center">
  <h2><b> SqLinear: A Linear Architecture for Large-Scale Traffic Prediction via Data-Adaptive Square Partitioning
 </b></h2>
</div>

<div align="center">

</div>

<div align="center">


</div>

<div align="center">

</div>

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 


## Contributions
**High-Quality Partitioning.** We propose a novel spatial partitioning method that generates balanced, non-overlapping, and geometrically-regular patches without padding requirements, establishing an optimal foundation for large-scale traffic prediction.

**Hierarchical Linear Modling.** We develop a hierarchical linear interaction module that efficiently captures both inter-patch and intra-patch spatio-temporal interactions, reducing the quadratic complexity bottleneck of attention-based approaches to linear complexity while maintaining modeling fidelity.    

**Theoretical Analysis.** We provide rigorous theoretical guarantees for our partitioning method, proving its effectiveness in preserving network topology and ensuring efficiency.

**Extensive Experiments.** An experimental study on 4 large-scale datasets shows that SqLinear achieves the state-of-the-art prediction accuracy while reducing parameter counts by $2\times$ and accelerating training by $3\times$ compared to existing baselines, validating its practical utility for city-scale deployment.
<p align="center">
<img src="./imgs/patching.png"  width="600" alt="" align=center />
</p>

## Requirements
- torch==1.11.0
- timm==1.0.12
- scikit_learn==1.0.2
- tqdm==4.67.1
- pandas==1.4.1
- numpy==1.22.3

## Folder Structure

```tex
└── code-and-data
    ├── config                 # Including detail configurations
    ├── cpt                    # Storing pre-trained weight files (manually create the folder and download files)
    ├── data                   # Including traffic data (download), adj files (generated), and the meta data
    ├── lib
    │   |──  utils.py          # Codes of preprocessing datasets and calculating metrics
    ├── log                    # Storing log files
    ├── model
    │   |──  models.py         # The core source code of our PatchSTG
    ├── main.py                # This is the main file for training and testing
    └── README.md              # This document
```

## Datasets
You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/1BDH1C66BCKBe7ge8G-rBaj1j3p0iR0TC?usp=sharing), then place the downloaded contents under the correspond dataset folder such as `./data/SD`.

## PreTrained Weights
You can access the pretrained weights from [[Google Drive]](https://drive.google.com/drive/folders/1hFyV2C10P3wl3OJkNNhhHb2LTKXcJ2mO?usp=sharing), then place the downloaded contents under the constructed cpt folder `./cpt`.

## Quick Demos
1. Download datasets and place them under `./data`
2. We provide pre-trained weights of results in the paper and the detail configurations under the folder `./config`. For example, you can test the SD dataset by:

```
python main.py --config ./config/SD.conf
```

3. If you want to train the model yourself, you can use the code at line 262 of the main file.

