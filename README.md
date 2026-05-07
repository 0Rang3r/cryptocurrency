# Cryptocurrency Market Prediction

This repository contains the experimental code and result files for my master's thesis project on cryptocurrency market prediction.

The project focuses on short-term BTC/USDT price direction prediction. The main task is to predict whether the BTC/USDT closing price will increase within the next 4 hours by using market trading features and sentiment-related statistical features.

## Project Overview

This project studies whether multimodal information can improve short-term cryptocurrency market prediction.

The experiment combines two types of features:

- market features, such as price, volume, return, and volatility;
- sentiment and information-flow features, such as FOMO, FUD, Euphoria, Panic, and information intensity.

The project compares several models, from a simple baseline model to more complex adaptive and emotion-enhanced models.

This repository is mainly used to store the code, data processing scripts, experiment configurations, result tables, and figures used in the thesis.

## Prediction Task

The prediction task is a binary classification problem.

The model uses historical market and sentiment features to predict the BTC/USDT price direction after 4 hours.

The target variable is defined as:

```text
1: the future closing price is higher than the current closing price
0: the future closing price is not higher than the current closing price
```

This project is not intended to build a complete trading system. It is used for academic analysis of short-term price direction prediction.

## Data

The main dataset is:

```text
btc_multimodal_hourly_dataset.csv
```

The dataset contains hourly BTC/USDT records.

Main market features include:

```text
open, high, low, close, volume, return, Vol_t
```

Main sentiment-related features include:

```text
Info_t, FOMO_m, FUD_m, Euphoria_m, Panic_m
```

The prediction horizon is 4 hours.

## Models

The experiment compares five models:

| Model | Description |
|---|---|
| M1 | Baseline LSTM using market features |
| M2 | Vanilla Transformer |
| M3 | Fixed-window model with emotion attention |
| M4 | Adaptive-window Transformer |
| M5 | Adaptive emotion-enhanced model with teacher-student training |

The comparison is designed to evaluate the effect of:

- adding sentiment features;
- using fixed and adaptive time windows;
- introducing emotion-enhanced representations;
- applying teacher-student training.

## Main Results

The main experiment results are stored in:

```text
ablation_results_teacher_m5_m5plus.csv
```

Main figures include:

```text
fig0_results_summary_table.png
fig1_metrics_comparison_bar_m5plus.png
fig2_confusion_matrices_m5plus.png
fig3_enhanced_roc_m5plus.png
fig4_trading_backtest.png
fig5_temporal_representation_similarity_heatmap.png
fig6_training_curves.png
```

According to the experimental results, the M5 model achieves the best overall performance among the compared models. Its advantage is mainly reflected in Recall, F1-score, AUC, and Accuracy.

However, the absolute prediction quality is still limited. Therefore, the model should be understood as an experimental forecasting method rather than a ready-to-use trading strategy.

## Repository Structure

The repository contains the following types of files:

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── btc_multimodal_hourly_dataset.csv
├── src/
│   ├── crypto_predictor/
│   ├── dataset_loader.py
│   ├── dataset_adaptive.py
│   ├── dataset_adaptive_m5plus.py
│   └── build_cache_adaptive.py
├── experiments/
│   ├── run_all_experiments_main.py
│   ├── run_stability_m4_m5.py
│   └── experiment configuration files
├── results/
│   ├── ablation_results_teacher_m5_m5plus.csv
│   ├── stability result files
│   └── backtest result files
└── figures/
    ├── result summary figures
    ├── confusion matrices
    ├── ROC curve
    ├── backtest figure
    └── training curves
```

The actual structure may be slightly different depending on how the files are organized.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Main dependencies include:

```text
numpy
pandas
scikit-learn
matplotlib
torch
tqdm
```

## Running the Experiments

To run the main experiment:

```bash
python run_all_experiments_main.py
```

To run the stability experiment for M4 and M5:

```bash
python run_stability_m4_m5.py
```

Some intermediate cache files may need to be generated before training, depending on the local file structure.

## Notes

- This project is prepared for academic research.
- The backtest is included only as an additional evaluation of predictive usefulness.
- This project does not provide investment advice.
- The model results should not be interpreted as a production-level trading system.

## Author

This project was prepared as part of a master's thesis on multimodal cryptocurrency market prediction.
