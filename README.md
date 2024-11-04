# CSF-net: Cross-Modal Spatiotemporal Fusion Network for Pulmonary Nodule Malignancy Predicting

![](https://img.shields.io/badge/-Github-181717?style=flat-square&logo=Github&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Awesome-FC60A8?style=flat-square&logo=Awesome&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=Python&logoColor=FFFFFF)
![](https://img.shields.io/badge/-Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=FFFFFF)

## Proposed method
This code is a pytorch implementation of our paper "CSF-net: Cross-Modal Spatiotemporal Fusion Network for Pulmonary Nodule Malignancy Predicting".
It consists of three components: (a) spatial feature extraction module, (b) temporal residual fusion module, and (c) CMAF module.The spatial module combines ResNet with the CBAM for enhanced feature representation. The temporal residual fusion module integrates features from different time points to capture correlations. The CMAF module uses cross-modal attention to effectively integrate follow-up and clinical data.The figure below shows our proposed network.

![image](model.png)

## The Temporal Residual Fusion module in our method
![image](tepfusion.png)
