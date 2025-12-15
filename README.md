## [ICASSP 2026] WPKAN-Mixer: Long-Term Time Series Forecasting with Adaptive Function Learning on Wavelet-Decomposed Components

> **Authors:**

Lei Liu, Guoqing Tang, Hongwei Zhao, Tengyuan Liu, Jiahui Huang, Bin Li.

<!-- This repo contains the code and data from our paper published in Journal of Power Sources [A Novel Patch-Based Transformer for Accurate Remaining Useful Life Prediction of Lithium-Ion Batteries](Paper: https://www.sciencedirect.com/science/article/pii/S0378775325000230). -->

## 1. Abstract

Time series forecasting is widely applied in fields like weather, power, and traffic management, but the complex multi-scale dynamics of real-world sequences make accurate forecasting extremely challenging. Existing methods typically employ effective decomposition techniques for time series forecasting tasks, while these approaches remain subject to inherent limitations: their modeling units apply a unified representation paradigm to all components, which hinders the adaptive modeling of local nonlinear dynamics across components at multiple scales, thus limiting fitting accuracy. This paper introduces WPKAN-Mixer for long-term time series forecasting, comprising primarily a Wavelet Transform module, a Patching-Embedding module and a dual Kolmogorov-Arnold Networks (KAN) Mixers module. We leverage the time-frequency localization of the wavelet transform module to achieve fine-grained temporal-signal decomposition. Furthermore, we design a dual KAN Mixers module incorporating the spline-based learnable function space to ``tailor-make'' an optimal functional representation for each frequency component, enabling flexible capture of non-linear dynamics characteristics from both temporal and feature dimensions. Experiments show that WPKAN-Mixer significantly outperforms state-of-the-art models on public benchmarks, achieving an average MSE and MAE reduction of 2-3% on most datasets.

## 2. Requirements

```bash
torch==2.3.1
optuna==3.6.1
numpy==1.25.2
pandas==1.5.3
matplotlib==3.7.2
tabulate==0.9.0
PyWavelets==1.6.0
scikit-learn==1.3.0
ikan
```

## 3. Datasets

You can download the datasets used in our experiments from: https://github.com/kwuking/TimeMixer

The data preprocessing code is provided.

## 4. Usage

- an example for train and evaluate a new model：

```bash
bash ETTm1_pl96.sh
```

- You can get the following output log:

```bash
Start Training- WPKANMixer_ETTm1_dec-True_sl96_pl96_dm256_bt64_wvdb3_tf3_df8_ptl16_stl4_sd42
train 34369
val 11425
test 11425
Epoch 1: cost time: 194.05 sec
    Epoch 1: Steps- 537 | Train Loss: 0.13445 Vali.MSE: 0.39833 Vali.MAE: 0.41592 Test.MSE: 0.32211 Test.MAE: 0.35736
    Validation loss decreased (inf --> 0.398334).  Saving model ...
Epoch 2: cost time: 193.06 sec
    Epoch 2: Steps- 537 | Train Loss: 0.11807 Vali.MSE: 0.40326 Vali.MAE: 0.41879 Test.MSE: 0.31641 Test.MAE: 0.35530
    EarlyStopping counter: 1 out of 5
Updating learning rate to 5e-05
Epoch 3: cost time: 193.51 sec
    Epoch 3: Steps- 537 | Train Loss: 0.11066 Vali.MSE: 0.39408 Vali.MAE: 0.41074 Test.MSE: 0.30895 Test.MAE: 0.34637
    Validation loss decreased (0.398334 --> 0.394078).  Saving model ...
Epoch 4: cost time: 192.94 sec
    Epoch 4: Steps- 537 | Train Loss: 0.10963 Vali.MSE: 0.39393 Vali.MAE: 0.41076 Test.MSE: 0.30934 Test.MAE: 0.34643
    Validation loss decreased (0.394078 --> 0.393931).  Saving model ...
Updating learning rate to 1e-05
Epoch 5: cost time: 193.42 sec
    Epoch 5: Steps- 537 | Train Loss: 0.10887 Vali.MSE: 0.39480 Vali.MAE: 0.41098 Test.MSE: 0.30950 Test.MAE: 0.34655
    EarlyStopping counter: 1 out of 5
Epoch 6: cost time: 193.51 sec
    Epoch 6: Steps- 537 | Train Loss: 0.10873 Vali.MSE: 0.39616 Vali.MAE: 0.41198 Test.MSE: 0.30995 Test.MAE: 0.34688
    EarlyStopping counter: 2 out of 5
Updating learning rate to 5e-06
Epoch 7: cost time: 194.03 sec
    Epoch 7: Steps- 537 | Train Loss: 0.10865 Vali.MSE: 0.39568 Vali.MAE: 0.41137 Test.MSE: 0.30979 Test.MAE: 0.34681
    EarlyStopping counter: 3 out of 5
Epoch 8: cost time: 193.39 sec
    Epoch 8: Steps- 537 | Train Loss: 0.10853 Vali.MSE: 0.39454 Vali.MAE: 0.41053 Test.MSE: 0.30962 Test.MAE: 0.34663
    EarlyStopping counter: 4 out of 5
Updating learning rate to 1e-06
Epoch 9: cost time: 194.09 sec
    Epoch 9: Steps- 537 | Train Loss: 0.10845 Vali.MSE: 0.39344 Vali.MAE: 0.41035 Test.MSE: 0.30995 Test.MAE: 0.34671
    Validation loss decreased (0.393931 --> 0.393438).  Saving model ...
Epoch 10: cost time: 194.45 sec
    Epoch 10: Steps- 537 | Train Loss: 0.10852 Vali.MSE: 0.39452 Vali.MAE: 0.41080 Test.MSE: 0.30933 Test.MAE: 0.34636
    EarlyStopping counter: 1 out of 5
Updating learning rate to 5e-07
Start Testing- WPKANMixer_ETTm1_dec-True_sl96_pl96_dm256_bt64_wvdb3_tf3_df8_ptl16_stl4_sd42
test 11425
mse: 0.3099547326564789, mae: 0.3467114269733429
```

## 5. Acknowledgments

We appreciate the following open-sourced repositories for their valuable code base:

- [Time-series-Library](https://github.com/thuml/Time-Series-Library)
- [WPMixer](https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer)
- [TimeMixer](https://github.com/kwuking/TimeMixer)
- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [ikan](https://github.com/kwuking/ikan)

## 6. Citation

If you find our work useful in your research, please consider citing:



<!-- ```latex
@article{liu2025patchformer,
  title={PatchFormer: A novel patch-based transformer for accurate remaining useful life prediction of lithium-ion batteries},
  author={Liu, Lei and Huang, Jiahui and Zhao, Hongwei and Li, Tianqi and Li, Bin},
  journal={Journal of Power Sources},
  volume={631},
  pages={236187},
  year={2025},
  publisher={Elsevier}
}
``` -->

If you have any problems, contact me via liulei13@ustc.edu.cn.


