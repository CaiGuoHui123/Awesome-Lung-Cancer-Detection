# Medical AI for Early Detection of Lung Cancer: A Survey

Authors: [Guohui Cai](https://github.com/CaiGuoHui123), [Ying Cai](https://ieeexplore.ieee.org/author/37087137422)*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/), [Yuanzhouhan Cao](https://scholar.google.com/citations?hl=en&user=-RBi2JcAAAAJ&view_op=list_works&sortby=pubdate), Lin Wu, [Daji Ergu](https://ieeexplore.ieee.org/author/37085795653), [Zhinbin Liao](https://researchers.adelaide.edu.au/profile/zhibin.liao#), [Yang Zhao](https://yangyangkiki.github.io/)

*Corresponding author

[[**Paper Link**](https://arxiv.org/abs/2410.14769)] [[Papers With Code]()]

![img](https://github.com/user-attachments/assets/584875dd-2db6-4b97-8b2d-245ef0b801a8)

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
- Global cancer statistics 2022: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries https://doi.org/10.3322/caac.21660
- Machine learning techniques for pulmonary nodule computer-aided diagnosis using CT images: A systematic review https://doi.org/10.1016/j.bspc.2023.104104
- Imagenet classification with deep convolutional neural networks https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
- Analysis based on machine and deep learning techniques for the accurate detection of lung nodules from CT images https://doi.org/10.1016/j.bspc.2023.105055
- A Bird’s Eye View Approach on the Usage of Deep Learning Methods in Lung Cancer Detection and Future Directions Using X-Ray and CT Images https://doi.org/10.1007/s11831-023-09749-6
- Support-vector networks https://doi.org/10.1007/BF00994018
- Nearest neighbor pattern classification https://ieeexplore.ieee.org/document/1053964
- Extreme learning machine: theory and applications https://doi.org/10.1016/j.neucom.2005.12.126
- A training algorithm for optimal margin classifiers https://dl.acm.org/doi/10.1145/130385.130401
- Discriminatory analysis, nonparametric discrimination https://www.af.mil/Portals/1/documents/rr/51fix.pdf
- Image classification using convolutional neural networks and kernel extreme learning machines https://ieeexplore.ieee.org/document/8451535
- Learning representations by back-propagating errors https://doi.org/10.1038/323533a0
- Generative adversarial nets https://papers.nips.cc/paper/5423-generative-adversarial-nets
- A fast learning algorithm for deep belief nets https://doi.org/10.1162/neco.2006.18.7.1527
- Gradient-based learning applied to document recognition https://ieeexplore.ieee.org/document/726791
- Deep residual learning for image recognition https://ieeexplore.ieee.org/document/7780459
- Faster r-cnn: Towards real-time object detection with region proposal networks https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks
- U-net: Convolutional networks for biomedical image segmentation https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28
- You only look once: Unified, real-time object detection https://ieeexplore.ieee.org/document/7780460
- Ssd: Single shot multibox detector https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2
- Long short-term memory https://doi.org/10.1162/neco.1997.9.8.1735
- Learning phrase representations using RNN encoder-decoder for statistical machine translation https://arxiv.org/abs/1406.1078
- The lung image database consortium (LIDC) and image database resource initiative (IDRI): a completed reference database of lung nodules on CT scans https://doi.org/10.1118/1.3528204
- Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: the LUNA16 challenge https://doi.org/10.1016/j.media.2017.06.001
- Early lung cancer action project: a summary of the findings on baseline screening https://doi.org/10.1634/theoncologist.6-2-147
- Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach https://doi.org/10.1038/ncomms5006
- Comparing and combining algorithms for computer-aided detection of pulmonary nodules in computed tomography scans: the ANODE09 study https://doi.org/10.1016/j.media.2010.05.003
- An adaptive morphology based segmentation technique for lung nodule detection in thoracic CT image https://doi.org/10.1016/j.cmpb.2020.105720
- Light deep model for pulmonary nodule detection from CT scan images for mobile devices https://doi.org/10.1155/2020/8893494
- An effective neural network model for lung nodule detection in CT images with optimal fuzzy model https://doi.org/10.1007/s11042-020-08813-x
- Fusion of 3D lung CT and serum biomarkers for diagnosis of multiple pathological types on pulmonary nodules https://doi.org/10.1016/j.cmpb.2021.106381
- A novel deep learning framework for lung nodule detection in 3D CT images https://doi.org/10.1007/s11042-021-11608-1
- Automatic lung nodule detection in thoracic CT scans using dilated slice-wise convolutions https://doi.org/10.1002/mp.14902
- Deep convolutional neural networks for multiplanar lung nodule detection: Improvement in small nodule identification https://doi.org/10.1002/mp.14679
- Diagnosis of Pulmonary Nodules on CT Images Using YOLOv4 https://doi.org/10.3991/ijoe.v18i05.27939
- Lung nodule detection and classification from Thorax CT-scan using RetinaNet with transfer learning https://doi.org/10.1016/j.jksuci.2020.07.010
- Prediction and Classification of CT images for Early Detection of Lung Cancer Using Various Segmentation Models https://doi.org/10.23956/ijheer.v10i4.1027
- A novel deep learning model based on multi-scale and multi-view for detection of pulmonary nodules https://doi.org/10.1007/s10278-023-00655-6
- Auto diagnostic system for detecting solitary and juxtapleural pulmonary nodules in computed tomography images using machine learning https://doi.org/10.1007/s00521-022-07673-2
- Pulmonary lung nodule detection from computed tomography images using two-stage convolutional neural network https://doi.org/10.1093/comjnl/bxad045
- Pulmonary lung nodule detection and classification through image enhancement and deep learning https://doi.org/10.1504/IJBM.2023.126939
- An interpretable three-dimensional artificial intelligence model for computer-aided diagnosis of lung nodules in computed tomography images https://doi.org/10.3390/cancers15184655
- A novel deep learning approach for the detection and classification of lung nodules from CT images https://doi.org/10.1007/s11042-023-14890-z
- Nodule Detection Using Local Binary Pattern Features to Enhance Diagnostic Decisions https://doi.org/10.32604/cmc.2024.030682
- Wavelet U-Net++ for accurate lung nodule segmentation in CT scans: Improving early detection and diagnosis of lung cancer https://doi.org/10.1016/j.bspc.2023.105509
- MEDS-Net: Multi-encoder based self-distilled network with bidirectional maximum intensity projections fusion for lung nodule detection https://doi.org/10.1016/j.engappai.2023.107597
- DBPNDNet: dual-branch networks using 3DCNN toward pulmonary nodule detection https://doi.org/10.1007/s11517-023-02769-3
- Lung cancer detection from thoracic CT scans using an ensemble of deep learning models https://doi.org/10.1007/s00521-023-08237-x
- An early prediction and classification of lung nodule diagnosis on CT images based on hybrid deep learning techniques https://doi.org/10.1007/s11042-023-15347-6
- Mask R-CNN-based detection and segmentation for pulmonary nodule 3D visualization diagnosis https://doi.org/10.1109/ACCESS.2020.2977139
- A two-stage convolutional neural networks for lung nodule detection https://doi.org/10.1109/JBHI.2020.2965515
- CPM-Net: A 3D center-points matching network for pulmonary nodule detection in CT scans https://doi.org/10.1007/978-3-030-59713-9_54
- Functional-realistic CT image super-resolution for early-stage pulmonary nodule detection https://doi.org/10.1016/j.future.2020.10.019
- An efficient multi-path 3D convolutional neural network for false-positive reduction of pulmonary nodule detection https://doi.org/10.1007/s11548-021-02437-6
- Automatic detection of pulmonary nodules in CT images based on 3D Res-I network https://doi.org/10.1007/s00371-020-01965-w
- Two-stage lung nodule detection framework using enhanced UNet and convolutional LSTM networks in CT images https://doi.org/10.1016/j.compbiomed.2022.106059
- Pulmonary nodule detection based on multiscale feature fusion https://doi.org/10.1155/2022/8903037
- Pulmonary nodules detection based on multi-scale attention networks https://doi.org/10.1038/s41598-022-05468-1
- An attentive and adaptive 3D CNN for automatic pulmonary nodule detection in CT image https://doi.org/10.1016/j.eswa.2022.118672
- Handcrafted features can boost performance and data-efficiency for deep detection of lung nodules from CT imaging https://doi.org/10.1109/ACCESS.2023.3286927
- Deep learning for the detection of benign and malignant pulmonary nodules in non-screening chest CT scans https://doi.org/10.1038/s43856-023-00321-5
- LungSeek: 3D Selective Kernel residual network for pulmonary nodule diagnosis https://doi.org/10.1007/s00371-022-02572-5
- YOLO-MSRF for lung nodule detection https://doi.org/10.1016/j.bspc.2023.106318
- Robust deep learning from incomplete annotation for accurate lung nodule detection https://doi.org/10.1016/j.compbiomed.2023.108361
- S-Net: an S-shaped network for nodule detection in 3D CT images https://doi.org/10.1088/1361-6560/acbd38
- Hierarchical approach for pulmonary-nodule identification from CT images using YOLO model and a 3D neural network classifier https://doi.org/10.1007/s12194-023-00688-x
- TiCNet: Transformer in Convolutional Neural Network for Pulmonary Nodule Detection on CT Images https://doi.org/10.1007/s10278-023-00575-6
- An intelligent lung nodule segmentation framework for early detection of lung cancer using an optimized deep neural system https://doi.org/10.1007/s11042-023-13620-4
- Development of a modified 3D region proposal network for lung nodule detection in computed tomography scans: a secondary analysis of lung nodule datasets https://doi.org/10.1186/s40644-024-00561-8
- A lung dense deep convolution neural network for robust lung parenchyma segmentation https://doi.org/10.1109/ACCESS.2020.2991212
- LNCDS: A 2D-3D cascaded CNN approach for lung nodule classification, detection and segmentation https://doi.org/10.1016/j.bspc.2021.102527
- Three-stage segmentation of lung region from CT images using deep neural networks https://doi.org/10.1186/s12880-021-00635-w
- Two-stage multitask U-Net construction for pulmonary nodule segmentation and malignancy risk prediction https://doi.org/10.21037/qims-20-1303
- Automatic lung tumor segmentation from CT images using improved 3D densely connected UNet https://doi.org/10.1007/s11517-022-02627-9
- MESAHA-Net: Multi-encoders based self-adaptive hard attention network with maximum intensity projections for lung nodule segmentation in CT scan https://arxiv.org/abs/2304.01576
- Integrated deep learning and stochastic models for accurate segmentation of lung nodules from computed tomography images: a novel framework https://doi.org/10.1109/ACCESS.2023.3302113
- Multiscale lung nodule segmentation based on 3D coordinate attention and edge enhancement https://doi.org/10.3934/era.2024125
- A deep Residual U-Net convolutional neural network for automated lung segmentation in computed tomography images https://doi.org/10.1016/j.bbe.2020.03.010
- R2U3D: Recurrent residual 3D U-Net for lung segmentation https://doi.org/10.1109/ACCESS.2021.3080964
- HR-MPF: high-resolution representation network with multi-scale progressive fusion for pulmonary nodule segmentation and classification https://doi.org/10.1186/s13640-021-00573-w
- CSE-GAN: A 3D conditional generative adversarial network with concurrent squeeze-and-excitation blocks for lung nodule segmentation https://doi.org/10.1016/j.compbiomed.2022.105781
- Artificial intelligence aided diagnosis of pulmonary nodules segmentation and feature extraction https://doi.org/10.1016/j.crad.2023.03.007
- A bi-directional deep learning architecture for lung nodule semantic segmentation https://doi.org/10.1007/s00371-023-02753-w
- Pulmonary Nodule Segmentation Network Based on Res Select Kernel Contextual U-Net https://doi.org/10.1115/1.4054776
- MRUNet-3D: A multi-stride residual 3D UNet for lung nodule segmentation https://doi.org/10.1016/j.ymeth.2023.226089
- An improved V-Net lung nodule segmentation model based on pixel threshold separation and attention mechanism https://doi.org/10.1038/s41598-024-46542-3
- A dual-task region-boundary aware neural network for accurate pulmonary nodule segmentation https://doi.org/10.1016/j.jvcir.2023.103909
- MANet: Multi-branch attention auxiliary learning for lung nodule detection and segmentation https://doi.org/10.1016/j.cmpb.2023.107748
- Effective deep learning approach for segmentation of pulmonary cancer in thoracic CT image https://doi.org/10.1016/j.bspc.2024.105804
- Overcoming the Challenge of Accurate Segmentation of Lung Nodules: A Multi-crop CNN Approach https://doi.org/10.1007/s10278-024-00618-2
- A holistic deep learning approach for identification and classification of sub-solid lung nodules in computed tomographic scans https://doi.org/10.1016/j.compeleceng.2020.106626
- Lung nodule detection and classification based on geometric fit in parametric form and deep learning https://doi.org/10.1007/s00521-020-04820-8
- Detection and classification of pulmonary nodules using deep learning and swarm intelligence https://doi.org/10.1007/s11042-020-09186-3
- False positive reduction in pulmonary nodule classification using 3D texture and edge feature in CT images https://doi.org/10.3233/THC-202389
- A novel receptive field-regularized V-net and nodule classification network for lung nodule detection https://doi.org/10.1002/ima.22415
- Accurate classification of nodules and non-nodules from computed tomography images based on radiomics and machine learning algorithms https://doi.org/10.1002/ima.22498
- An approach for classification of lung nodules https://doi.org/10.1177/2633430721101264
- LDNNET: towards robust classification of lung nodule and cancer using lung dense neural network https://doi.org/10.1109/ACCESS.2021.3058917
- An uncertainty-aware self-training framework with consistency regularization for the multilabel classification of common computed tomography signs in lung nodules https://doi.org/10.21037/qims-23-252
- Artificial intelligence solution to classify pulmonary nodules on CT https://doi.org/10.1016/j.diii.2020.07.016
- Shape and margin-aware lung nodule classification in low-dose CT images via soft activation mapping https://doi.org/10.1016/j.media.2020.101628
- NROI based feature learning for automated tumor stage classification of pulmonary lung nodules using deep convolutional neural networks https://doi.org/10.1016/j.jksuci.2021.07.011
- Atrous convolution aided integrated framework for lung nodule segmentation and classification https://doi.org/10.1016/j.bspc.2023.104527
- Early detection and classification of malignant lung nodules from CT images: An optimal ensemble learning https://doi.org/10.1016/j.eswa.2023.120361
- Fuzzy information granulation towards benign and malignant lung nodules classification https://doi.org/10.1016/j.cmpbup.2023.100153
- Texture and radiomics inspired data-driven cancerous lung nodules severity classification https://doi.org/10.1016/j.bspc.2023.105543
- Towards reliable and explainable AI model for pulmonary nodule diagnosis https://doi.org/10.1016/j.bspc.2023.105646
- Multi-task learning for lung nodule classification on chest CT https://doi.org/10.1109/ACCESS.2020.3029659
- Pulmonary nodule classification using feature and ensemble learning-based fusion techniques https://doi.org/10.1109/ACCESS.2021.3108699
- 3D SAACNet with GBM for the classification of benign and malignant lung nodules https://doi.org/10.1016/j.compbiomed.2023.106532
- Detection and classification of lung cancer computed tomography images using a novel improved deep belief network with Gabor filters https://doi.org/10.1016/j.chemolab.2023.104763
- Automated 3D convolutional neural network architecture design using genetic algorithm for pulmonary nodule classification https://doi.org/10.11591/eei.v13i3.4344
- An efficient combined intelligent system for segmentation and classification of lung cancer computed tomography images https://doi.org/10.7717/peerj-cs.1802
- Combined model integrating deep learning, radiomics, and clinical data to classify lung nodules at chest CT https://doi.org/10.1007/s11547-023-01560-x
- A fast and efficient CAD system for improving the performance of malignancy level classification on lung nodules https://doi.org/10.1109/ACCESS.2020.2972235
- High-resolution CT image analysis based on 3D convolutional neural network can enhance the classification performance of radiologists in classifying pulmonary non-solid nodules https://doi.org/10.1016/j.ejrad.2021.109810
- Self-supervised transfer learning framework driven by visual attention for benign-malignant lung nodule classification on chest CT https://doi.org/10.1016/j.eswa.2022.119339
- Novel Algorithm for Pulmonary Nodule Classification using CNN on CT Scans https://doi.org/10.20549/ijisae.2024.0152
- EDICNet: An end-to-end detection and interpretable malignancy classification network for pulmonary nodules in computed tomography https://doi.org/10.1117/12.2548858
- Study on Identification Method of Pulmonary Nodules: Improved Random Walk Pulmonary Parenchyma Segmentation and Fusion Multi-Feature VGG16 Nodule Classification https://doi.org/10.3389/fonc.2022.822827
- Impact of localized fine tuning in the performance of segmentation and classification of lung nodules from computed tomography scans using deep learning https://doi.org/10.3389/fonc.2023.1140635
- Lung-EffNet: Lung cancer classification using EfficientNet from CT-scan images https://doi.org/10.1016/j.engappai.2023.106902
- No surprises: Training robust lung nodule detection for low-dose CT scans by augmenting with adversarial attacks https://doi.org/10.1109/TMI.2020.3025919
- Advancing pulmonary nodule diagnosis by integrating Engineered and Deep features extracted from CT scans https://doi.org/10.3390/algorithms17040161
- LGDNet: local feature coupling global representations network for pulmonary nodules detection https://doi.org/10.1007/s11517-023-02750-w
- Hessian-MRLoG: Hessian information and multi-scale reverse LoG filter for pulmonary nodule detection https://doi.org/10.1016/j.compbiomed.2020.104272
- Attention is all you need https://arxiv.org/abs/1706.03762
- Lung Nodule Segmentation and Uncertain Region Prediction with an Uncertainty-Aware Attention Mechanism https://doi.org/10.1109/TMI.2023.3291485
- Deeplung: Deep 3D dual path nets for automated pulmonary nodule detection and classification https://doi.org/10.1109/WACV.2018.00082
- Computer-aided diagnostic system kinds and pulmonary nodule detection efficacy https://doi.org/10.11591/ijece.v12i5.25355
- A novel method for lung nodule detection in computed tomography scans based on Boolean equations and vector of filters techniques https://doi.org/10.1016/j.compeleceng.2021.107911
- Development and validation of a modified three-dimensional U-Net deep-learning model for automated detection of lung nodules on chest CT images from the lung image database consortium and Japanese datasets https://doi.org/10.1016/j.acra.2021.01.001
- A two-stage framework for automated malignant pulmonary nodule detection in CT scans https://doi.org/10.3390/diagnostics10030131
- Detection and Classification of Lung Carcinoma using CT scans https://doi.org/10.1088/1742-6596/2286/1/012011
- A framework for understanding artificial intelligence research: insights from practice https://doi.org/10.1108/JEIM-01-2020-0006
- DS-MSFF-Net: Dual-path self-attention multi-scale feature fusion network for CT image segmentation https://doi.org/10.1007/s10489-023-04523-7
- Exploring pretrained encoders for lung nodule segmentation task using LIDC-IDRI dataset https://doi.org/10.1007/s11042-023-14815-1
- SM-RNet: A Scale-aware-based Multi-attention Guided Reverse Network for Pulmonary Nodules Segmentation https://doi.org/10.1109/TIM.2023.3313251
- MDFN: A Multi-level Dynamic Fusion Network with self-calibrated edge enhancement for lung nodule segmentation https://doi.org/10.1016/j.bspc.2024.105507
- Automatically segmenting and classifying the lung nodules from CT images https://doi.org/10.20549/ijisae.2024.0152
- Jointvit: Modeling oxygen saturation levels with joint supervision on long-tailed octa https://doi.org/10.1007/978-3-030-81807-3_13
- A deep learning approach to diabetes diagnosis https://doi.org/10.1007/978-3-030-81807-3_13
- BHSD: A 3D Multi-class Brain Hemorrhage Segmentation Dataset https://doi.org/10.1007/978-3-030-81807-3_13
- Segreg: Segmenting oars by registering mr images and ct annotations https://doi.org/10.1109/ISBI.2024.937389
- Segstitch: Multidimensional transformer for robust and efficient medical imaging segmentation https://arxiv.org/abs/2408.00496
- Xlip: Cross-modal attention masked modelling for medical language-image pre-training https://arxiv.org/abs/2407.19546
- Sine Activated Low-Rank Matrices for Parameter Efficient Learning https://arxiv.org/abs/2403.19243
- Thin-Thick Adapter: Segmenting Thin Scans Using Thick Annotations https://openreview.net/forum?id=r1x8a0Fzv
- Esa: Annotation-efficient active learning for semantic segmentation https://arxiv.org/abs/2408.13491
- A landmark-based approach for instability prediction in distal radius fractures https://doi.org/10.1109/ISBI.2024.938472
- Can rotational thromboelastometry rapidly identify theragnostic targets in isolated traumatic brain injury? https://doi.org/10.1111/eme.13735
- MedDet: Generative Adversarial Distillation for Efficient Cervical Disc Herniation Detection https://arxiv.org/abs/2409.00204
- MambaClinix: Hierarchical Gated Convolution and Mamba-Based U-Net for Enhanced 3D Medical Image Segmentation https://arxiv.org/abs/2409.12533
- VM-UNET-V2: rethinking vision mamba UNet for medical image segmentation https://doi.org/10.1007/978-3-030-81807-3_13
- MSDet: Receptive Field Enhanced Multiscale Detection for Tiny Pulmonary Nodule https://arxiv.org/abs/2409.14028
- Mamba: Linear-time sequence modeling with selective state spaces https://arxiv.org/abs/2312.00752
- Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality https://arxiv.org/abs/2405.21060
- Motion mamba: Efficient and long sequence motion generation with hierarchical and bidirectional selective ssm https://arxiv.org/abs/2403.07487
- InfiniMotion: Mamba Boosts Memory in Transformer for Arbitrary Long Motion Generation https://arxiv.org/abs/2407.10061
- Classification of non-small cell lung cancers using deep convolutional neural networks https://doi.org/10.1007/s11042-024-12809-8








