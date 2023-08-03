# HIV_IN_Classification_ML

## ***Novel machine learning approach toward classification model of HIV-1 integrase inhibitors***

**Abstract**
HIV-1 (Human immunodeficiency virus-1) has been causing severe pandemics by attacking the immune system of its host. Left untreated, it can lead to AIDS (acquired immunodeficiency syndrome), where death is inevitable due to opportunistic diseases. Therefore, discovering new antiviral drugs against HIV-1 is crucial. This study aimed to explore a novel machine learning approach to classify compounds that inhibit HIV-1 integrase and screen the dataset of repurposing compounds. The present study had two main stages: selecting the best type of fingerprint or molecular descriptor using the Wilcoxon signed-rank test and building a computational model based on machine learning. In the first stage, we calculated 16 different types of fingerprint or molecular descriptors from the dataset and used each of them as input features for 10 machine-learning models, which were evaluated through cross-validation. Then, a meta-analysis was performed with the Wilcoxon signed-rank test to select the optimal fingerprint or molecular descriptors types. In the second stage, we constructed a model based on the optimal fingerprint or molecular descriptor type. This data followed the machine learning procedure, including data preprocessing, outlier handling, normalization, feature selection, model selection, external validation, and model optimization. In the end, an XGBoost model and RDK7 fingerprint were identified to be the most suitable. The model achieved promising results, with an average precision of 0.928 ± 0.027 and an F1-score of 0.848 ± 0.041 in cross-validation. The model achieved an average precision of 0.921 and an F1-score of 0.889 in external validation. Molecular docking was performed and validated by redocking for docking power and retrospective control for screening power, with the AUC metrics being 0.876 and the threshold being identified at –9.71 kcal/mol. Finally, 44 compounds from DrugBank repurposing data were selected from the QSAR model, then three candidates were identified as potential compounds from molecular docking, and PSI-697 was detected as the most promising molecule, with in vitro experiment being not performed (docking score: -17.14 kcal/mol, HIV integrase inhibitory probability: 69.81%)

Keywords: machine learning, HIV-1, integrase, inhibitors, Wilcoxon, virtual screening

![screenshot](./Img/Abstract_graphic.png)

## Contributors
- [Tieu-Long Phan](https://tieulongphan.github.io/)
- [Hoang-Son Le Lai]()
- [The-Chuong Trinh](https://trinhthechuong.github.io/)
- [Gia-Bao Truong](https://github.com/buchijw)
- [Van-Thinh To](https://thinhump.github.io/)
- [Phuoc-Chung Nguyen Van](https://www.facebook.com/chung.nguyenvanphuoc.9)
- [Thanh-An Pham](https://github.com/anpham2209)
- [Ngoc-Tuyen Truong](https://scholar.google.com/citations?hl=vi&user=qx3eMsIAAAAJ) - [Corresponding author](mailto:truongtuyen@ump.edu.vn)
