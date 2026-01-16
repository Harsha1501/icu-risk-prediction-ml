Overview

This repository contains research code and experiments for uncertainty-aware machine learning models applied to ICU time-series data. The goal is to develop reliable, interpretable, and clinically meaningful risk prediction models that not only provide point estimates but also quantify predictive uncertainty, which is critical for real-world clinical decision-making.

The project focuses on early risk prediction tasks in the Intensive Care Unit (ICU), such as mortality or clinical deterioration, using multivariate physiological time-series data.

Research Motivation

Traditional deep learning models often provide highly confident predictions even when they are wrong. In safety-critical domains like healthcare, this can lead to dangerous clinical decisions.

This project addresses this issue by:

Incorporating uncertainty quantification into time-series models

Improving model reliability and calibration

Supporting risk-aware clinical decision-making

Key Research Questions

How can uncertainty be effectively modeled in ICU time-series prediction tasks?

Do uncertainty-aware models improve reliability compared to deterministic baselines?

How does predictive uncertainty correlate with data sparsity, noise, and patient heterogeneity?

Methods

The repository explores a range of machine learning and deep learning approaches, including:

Recurrent neural networks (LSTM / GRU)

Temporal convolutional models

Bayesian neural networks

Monte Carlo Dropout

Ensemble-based uncertainty estimation

Calibration techniques (e.g., reliability diagrams, ECE)

Evaluation focuses not only on predictive performance but also on uncertainty quality.

Datasets

Due to privacy, licensing, and size constraints, raw ICU datasets are not included in this repository.


Users must complete the required data use agreements and training to access these datasets.

Repository Structure
icu-risk-prediction-ml/
├── notebooks/          # Exploratory analysis and experiments
├── src/                # Model implementations and utilities
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
Reproducibility

To reproduce experiments:

pip install -r requirements.txt

Dataset preprocessing scripts expect data to be placed locally and are not tracked by Git.

Ethical Considerations

This project uses de-identified clinical data and follows best practices for:

Patient privacy

Responsible AI

Transparent uncertainty reporting

The code is intended for research purposes only and not for direct clinical deployment.

Intended Use

This repository is primarily intended for:

Academic research

Methodological development in uncertainty-aware ML

PhD-level research in machine learning for healthcare

Author

Harsha
Prospective PhD researcher in Machine Learning for Healthcare

License

This project is released for academic and research use. Dataset licenses are governed by their respective providers.
