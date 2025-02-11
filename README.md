# Medical AI for Early Detection of Lung Cancer: A Survey

Authors: [Guohui Cai](https://github.com/CaiGuoHui123), [Ying Cai](https://ieeexplore.ieee.org/author/37087137422)*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)†, [Yuanzhouhan Cao](https://scholar.google.com/citations?hl=en&user=-RBi2JcAAAAJ&view_op=list_works&sortby=pubdate), Lin Wu, [Daji Ergu](https://ieeexplore.ieee.org/author/37085795653), [Zhinbin Liao](https://researchers.adelaide.edu.au/profile/zhibin.liao#), [Yang Zhao](https://yangyangkiki.github.io/)

*Corresponding author. †Project lead.

[[**Paper Link**](https://arxiv.org/abs/2410.14769)] [[Papers With Code](https://paperswithcode.com/paper/medical-ai-for-early-detection-of-lung-cancer)]

![img](https://github.com/user-attachments/assets/584875dd-2db6-4b97-8b2d-245ef0b801a8)

## Citation
```
@article{cai2024medical,
  title={Medical ai for early detection of lung cancer: A survey},
  author={Cai, Guohui and Cai, Ying and Zhang, Zeyu and Cao, Yuanzhouhan and Wu, Lin and Ergu, Daji and Liao, Zhinbin and Zhao, Yang},
  journal={arXiv preprint arXiv:2410.14769},
  year={2024}
}
```

## Abstract
Lung cancer remains one of the leading causes of morbidity and mortality worldwide, making early diagnosis critical for improving therapeutic outcomes and patient prognosis. Computer-aided diagnosis (CAD) systems, which analyze CT images, have proven effective in detecting and classifying pulmonary nodules, significantly enhancing the detection rate of early-stage lung cancer. Although traditional machine learning algorithms have been valuable, they exhibit limitations in handling complex sample data. The recent emergence of deep learning has revolutionized medical image analysis, driving substantial advancements in this field. This review focuses on recent progress in deep learning for pulmonary nodule detection, segmentation, and classification. Traditional machine learning methods, such as SVM and KNN, have shown limitations, paving the way for advanced approaches like Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and Generative Adversarial Networks (GAN). The integration of ensemble models and novel techniques is also discussed, emphasizing the latest developments in lung cancer diagnosis. Deep learning algorithms, combined with various analytical techniques, have markedly improved the accuracy and efficiency of pulmonary nodule analysis, surpassing traditional methods, particularly in nodule classification. Although challenges remain, continuous technological advancements are expected to further strengthen the role of deep learning in medical diagnostics, especially for early lung cancer detection and diagnosis.

## Project Overview
This project focuses on the application of deep learning techniques to the detection, segmentation, and classification of pulmonary nodules in CT images, particularly for early-stage lung cancer detection. The methods leverage advanced neural networks such as Convolutional Neural Networks (CNNs), U-Nets, and their variants to improve diagnostic accuracy, reduce false positives, and enhance the overall sensitivity of Computer-Aided Diagnosis (CAD) systems.

The project is built upon two prominent datasets: **LIDC-IDRI** and **LUNA16**, both of which are publicly available and widely used in lung nodule research. By utilizing these datasets, the project aims to achieve a more comprehensive analysis of the performance of deep learning models in the medical imaging field.

## Datasets
The project utilizes several key datasets that have been essential in driving advancements in lung nodule detection and diagnosis:

- **LIDC-IDRI**: A large-scale dataset containing over 1,000 lung CT cases with multi-radiologist annotations. This dataset serves as the backbone for many of the studies in lung nodule detection.
- **LUNA16**: Derived from LIDC-IDRI, LUNA16 focuses on nodule detection, providing 888 high-quality CT scans with standardized metrics for evaluating algorithms.
- **Additional Datasets**: Datasets such as ELCAP, NSCLC, and ANODE09 are also referenced to supplement research efforts.

## Key Techniques and Models
This project explores multiple deep learning models tailored to different aspects of lung nodule detection, segmentation, and classification:

### Detection Models
- **CNN-based detection**: A variety of CNN architectures are used, ranging from lightweight models like Light CNN to more advanced networks such as U-Net++ and EfficientNet.
- **YOLOv8**: A real-time detection model, effective for fast nodule identification with minimal false positives.
- **Hybrid Approaches**: Fusion of 3D imaging techniques and biomarker data to enhance detection precision.

### Segmentation Models
- **U-Net Variants**: Models like Wavelet U-Net++ and 3D DenseUNet provide robust segmentation of nodules with a focus on improving Dice Similarity Coefficient (DSC).
- **Attention Mechanisms**: Self-attention networks (e.g., HSNet) improve segmentation accuracy by focusing on relevant features in CT scans.
- **Multi-task Learning**: Approaches that integrate both nodule detection and segmentation into a single pipeline for better performance.

### Classification Models
- **Deep Learning Classifiers**: CNN-based classifiers, such as those built on ResNet and DenseNet architectures, are employed to distinguish between benign and malignant nodules.
- **Ensemble Learning**: Hybrid deep learning models that combine multiple classifiers to enhance accuracy, sensitivity, and specificity.
- **SVM and Traditional Approaches**: For comparative purposes, traditional machine learning methods such as Support Vector Machines (SVM) are also tested against deep learning models.

## Performance Metrics
The performance of the models is evaluated using the following key metrics:
- **Sensitivity (True Positive Rate)**: Measures the model's ability to correctly identify nodules.
- **Specificity (True Negative Rate)**: Measures the model's ability to correctly identify non-nodules.
- **Dice Similarity Coefficient (DSC)**: Used in segmentation tasks to evaluate the overlap between predicted and ground truth nodule regions.
- **Area Under the Curve (AUC)**: Commonly used in classification tasks to evaluate the overall model performance.
- **Competition Performance Metric (CPM)**: Specific to lung nodule detection, measuring sensitivity at varying false-positive rates.

## Future Work
This project highlights the potential of deep learning models in improving the accuracy of lung nodule detection and classification. However, future work will focus on:
- **Improving interpretability**: Developing models that can provide insights into their decision-making process.
- **Real-time applications**: Enhancing the computational efficiency of models to allow for real-time diagnostic use in clinical settings.
- **Multimodal approaches**: Integrating clinical data, genomic information, and imaging data to improve diagnosis accuracy.

## Contributors
- Guohui Cai: Conceptualization, Investigation, Writing – original draft
- Ying Cai: Investigation, Writing – original draft
- Zeyu Zhang：Writing – review & editing
- Yuanzhouhan Cao: Methodology, Writing – review & editing
- Lin Wu: Investigation, Methodology
- Daji Ergu: Supervision
- Zhibin Liao: Validation
- Yang Zhao: Investigation, Supervision

## Acknowledgements
This research has been supported by the National Natural Science Foundation of China (Grant No. 72174172) and the Scientific and Technological Innovation Team for Qinghai-Tibetan Plateau Research at Southwest Minzu University (Grant No. 2024CXTD20). We sincerely appreciate their valuable support, which made this work possible.

## Key Papers
- Global cancer statistics 2022: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries https://doi.org/10.3322/caac.21834
- Imagenet classification with deep convolutional neural networks https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
- Support-vector networks https://doi.org/10.1007/BF00994018
- Nearest neighbor pattern classification https://ieeexplore.ieee.org/document/1053964
- Extreme learning machine: theory and applications https://doi.org/10.1016/j.neucom.2005.12.126
- A training algorithm for optimal margin classifiers https://dl.acm.org/doi/10.1145/130385.130401
- Image classification using convolutional neural networks and kernel extreme learning machines https://ieeexplore.ieee.org/document/8451560
- Learning representations by back-propagating errors https://doi.org/10.1038/323533a0
- A fast learning algorithm for deep belief nets https://doi.org/10.1162/neco.2006.18.7.1527
- Gradient-based learning applied to document recognition https://ieeexplore.ieee.org/document/726791
- Deep residual learning for image recognition https://ieeexplore.ieee.org/document/7780459
- U-net: Convolutional networks for biomedical image segmentation https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28
- You only look once: Unified, real-time object detection https://ieeexplore.ieee.org/document/7780460
- Ssd: Single shot multibox detector https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2
- Long short-term memory https://doi.org/10.1162/neco.1997.9.8.1735
- Learning phrase representations using RNN encoder-decoder for statistical machine translation https://arxiv.org/abs/1406.1078
- The lung image database consortium (LIDC) and image database resource initiative (IDRI): a completed reference database of lung nodules on CT scans https://doi.org/10.1118/1.3528204
- Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: the LUNA16 challenge https://doi.org/10.1016/j.media.2017.06.015
- Early lung cancer action project: a summary of the findings on baseline screening https://doi.org/10.1634/theoncologist.6-2-147
- Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach https://doi.org/10.1038/ncomms5006
- Comparing and combining algorithms for computer-aided detection of pulmonary nodules in computed tomography scans: the ANODE09 study https://doi.org/10.1016/j.media.2010.05.005
- An adaptive morphology based segmentation technique for lung nodule detection in thoracic CT image https://doi.org/10.1016/j.cmpb.2020.105720
- Light deep model for pulmonary nodule detection from CT scan images for mobile devices https://doi.org/10.1155/2020/8893494
- Fusion of 3D lung CT and serum biomarkers for diagnosis of multiple pathological types on pulmonary nodules https://doi.org/10.1016/j.cmpb.2021.106381
- Automatic lung nodule detection in thoracic CT scans using dilated slice-wise convolutions https://doi.org/10.1002/mp.14915
- Deep convolutional neural networks for multiplanar lung nodule detection: Improvement in small nodule identification https://doi.org/10.1002/mp.14648
- Pulmonary lung nodule detection from computed tomography images using two-stage convolutional neural network https://doi.org/10.1093/comjnl/bxab191
- Pulmonary lung nodule detection and classification through image enhancement and deep learning https://doi.org/10.1504/IJBM.2023.130637
- An interpretable three-dimensional artificial intelligence model for computer-aided diagnosis of lung nodules in computed tomography images https://doi.org/10.3390/cancers15184655
- Wavelet U-Net++ for accurate lung nodule segmentation in CT scans: Improving early detection and diagnosis of lung cancer https://doi.org/10.1016/j.bspc.2023.105509
- MEDS-Net: Multi-encoder based self-distilled network with bidirectional maximum intensity projections fusion for lung nodule detection https://doi.org/10.1016/j.engappai.2023.107597
- DBPNDNet: dual-branch networks using 3DCNN toward pulmonary nodule detection https://link.springer.com/article/10.1007/s11517-023-02957-1
- Two-stage lung nodule detection framework using enhanced UNet and convolutional LSTM networks in CT images https://doi.org/10.1016/j.compbiomed.2022.106059
- Pulmonary nodule detection based on multiscale feature fusion https://doi.org/10.1155/2022/8903037
- An attentive and adaptive 3D CNN for automatic pulmonary nodule detection in CT image https://doi.org/10.1016/j.eswa.2022.118672
- LNCDS: A 2D-3D cascaded CNN approach for lung nodule classification, detection and segmentation https://doi.org/10.1016/j.bspc.2021.102527
- MESAHA-Net: Multi-encoders based self-adaptive hard attention network with maximum intensity projections for lung nodule segmentation in CT scan https://arxiv.org/abs/2304.01576
- CSE-GAN: A 3D conditional generative adversarial network with concurrent squeeze-and-excitation blocks for lung nodule segmentation https://doi.org/10.1016/j.compbiomed.2022.105781
- A dual-task region-boundary aware neural network for accurate pulmonary nodule segmentation https://doi.org/10.1016/j.jvcir.2023.103909
- MANet: Multi-branch attention auxiliary learning for lung nodule detection and segmentation https://doi.org/10.1016/j.cmpb.2023.107748
- A holistic deep learning approach for identification and classification of sub-solid lung nodules in computed tomographic scans https://doi.org/10.1016/j.compeleceng.2020.106626
- Early detection and classification of malignant lung nodules from CT images: An optimal ensemble learning https://doi.org/10.1016/j.eswa.2023.120361
- Texture and radiomics inspired data-driven cancerous lung nodules severity classification https://doi.org/10.1016/j.bspc.2023.105543
- Towards reliable and explainable AI model for pulmonary nodule diagnosis https://doi.org/10.1016/j.bspc.2023.105646
- Detection and classification of lung cancer computed tomography images using a novel improved deep belief network with Gabor filters https://doi.org/10.1016/j.chemolab.2023.104763
- An efficient combined intelligent system for segmentation and classification of lung cancer computed tomography images https://doi.org/10.7717/peerj-cs.1802
- High-resolution CT image analysis based on 3D convolutional neural network can enhance the classification performance of radiologists in classifying pulmonary non-solid nodules https://doi.org/10.1016/j.ejrad.2021.109810
- Self-supervised transfer learning framework driven by visual attention for benign-malignant lung nodule classification on chest CT https://doi.org/10.1016/j.eswa.2022.119339
- Impact of localized fine tuning in the performance of segmentation and classification of lung nodules from computed tomography scans using deep learning https://doi.org/10.3389/fonc.2023.1140635
- Lung-EffNet: Lung cancer classification using EfficientNet from CT-scan images https://doi.org/10.1016/j.engappai.2023.106902
- Attention is all you need https://arxiv.org/abs/1706.03762
- A two-stage framework for automated malignant pulmonary nodule detection in CT scans https://doi.org/10.3390/diagnostics10030131
- Detection and Classification of Lung Carcinoma using CT scans https://doi.org/10.1088/1742-6596/2286/1/012011
- Segreg: Segmenting oars by registering mr images and ct annotations https://ieeexplore.ieee.org/abstract/document/10635437
- Segstitch: Multidimensional transformer for robust and efficient medical imaging segmentation https://arxiv.org/abs/2408.00496
- Xlip: Cross-modal attention masked modelling for medical language-image pre-training https://arxiv.org/abs/2407.19546
- Sine Activated Low-Rank Matrices for Parameter Efficient Learning https://arxiv.org/abs/2403.19243
- Esa: Annotation-efficient active learning for semantic segmentation https://arxiv.org/abs/2408.13491
- MedDet: Generative Adversarial Distillation for Efficient Cervical Disc Herniation Detection https://arxiv.org/abs/2409.00204
- MambaClinix: Hierarchical Gated Convolution and Mamba-Based U-Net for Enhanced 3D Medical Image Segmentation https://arxiv.org/abs/2409.12533
- MSDet: Receptive Field Enhanced Multiscale Detection for Tiny Pulmonary Nodule https://arxiv.org/abs/2409.14028
- Mamba: Linear-time sequence modeling with selective state spaces https://arxiv.org/abs/2312.00752
- Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality https://arxiv.org/abs/2405.21060
- Motion mamba: Efficient and long sequence motion generation with hierarchical and bidirectional selective ssm https://arxiv.org/abs/2403.07487
- InfiniMotion: Mamba Boosts Memory in Transformer for Arbitrary Long Motion Generation https://arxiv.org/abs/2407.10061

