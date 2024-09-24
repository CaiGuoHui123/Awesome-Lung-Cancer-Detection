# AI and Deep Learning Techniques for Early Detection, Diagnosis, and Classification of Lung Cancer: A Survey

Authors: [Guohui Cai](https://github.com/CaiGuoHui123), [Ying Cai](https://ieeexplore.ieee.org/author/37087137422)*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/), [Daji Ergu](https://ieeexplore.ieee.org/author/37085795653),
[Lin Wu], [Zhinbin Liao](https://researchers.adelaide.edu.au/profile/zhibin.liao#), Binbin Hu, [Yang Zhao](https://yangyangkiki.github.io/)

*Corresponding author

[[**Paper Link**]()] [[Papers With Code]()]

![img](https://github.com/user-attachments/assets/584875dd-2db6-4b97-8b2d-245ef0b801a8)

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
- Zeyu Zhang：Investigation, Writing – review & editing
- Daji Ergu: Supervision
- Lin Wu: Investigation, Methodology
- Zhibin Liao: Validation
- Binbin Hu: Methodology
- Yang Zhao: Investigation, Supervision

## Acknowledgements
This research has been supported by the National Natural Science Foundation of China (Grant No. 72174172) and the Scientific and Technological Innovation Team for Qinghai-Tibetan Plateau Research at Southwest Minzu University (Grant No. 2024CXTD20). We sincerely appreciate their valuable support, which made this work possible.
