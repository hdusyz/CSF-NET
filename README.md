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

## Experiment result
We compared our results with other state-of-the-art methods, and our results were better than any other comparison method. The results indicate that the introduced CMAF module can effectively capture the intrinsic correlation between the two modalities. The time residual module we designed has good sensitivity to features at different times, and these two modules significantly improve the performance of the model.

<table>
<thead>
  <tr>
    <th>Methods</th>
    <th>Acc</th>
    <th>Prec</th>
    <th>F1 score</th>
    <th>AUC</th>
    <th>Rec</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>SCANs</td>
    <td>0.7865</td>
    <td>0.7667</td>
    <td>0.7077</td>
    <td>0.7725</td>
    <td>0.6571</td>
  </tr>
  <tr>
    <td>NAS-Lung</td>
    <td>0.8539</td>
    <td>0.8235</td>
    <td>0.8116</td>
    <td>0.8910</td>
    <td>0.8000</td>
  </tr>
  <tr>
    <td>T-LSTM</td>
    <td>0.7645</td>
    <td>0.7012</td>
    <td>0.6527</td>
    <td>0.7778</td>
    <td>0.6000</td>
  </tr>
  <tr>
    <td>DeepCAD</td>
    <td>0.8590</td>
    <td>0.7879</td>
    <td>0.8254</td>
    <td>0.8990</td>
    <td>0.8667</td>
  </tr>
  <tr>
    <td>MFCN</td>
    <td>0.7949</td>
    <td>0.7059</td>
    <td>0.7500</td>
    <td>0.8903</td>
    <td>0.8000</td>
  </tr>
  <tr>
    <td>RadFusion</td>
    <td>0.7753</td>
    <td>0.8026</td>
    <td>0.6667</td>
    <td>0.7693</td>
    <td>0.6000</td>
  </tr>
  <tr>
    <td>CSF-Net (ours)</td>
    <td>0.8974</td>
    <td>0.8235</td>
    <td>0.8750</td>
    <td>0.9389</td>
    <td>0.9333</td>
  </tr>  
</tbody>
</table>
