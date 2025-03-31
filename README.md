# Fraud-Detection
Financial Fraud Detection System: Machine learning models to identify fraudulent transactions with 99.5% recall and zero false positives. Includes feature engineering, model training, and detailed analysis of fraud patterns.

# Financial Fraud Detection System

![Fraud Detection](https://img.shields.io/badge/ML-Fraud%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning approach to detect fraudulent financial transactions with high accuracy and minimal false alarms.

## 🔍 The Problem

In the financial world, fraudulent transactions cost institutions billions every year. Traditional rule-based systems struggle to keep up with evolving fraud tactics. This project creates a robust machine learning solution that:

- Detects 99.57% of fraudulent transactions
- Generates zero false positives
- Identifies key patterns that indicate fraud
- Provides actionable insights for fraud prevention

## 📊 The Dataset

The dataset contains millions of financial transactions with the following characteristics:
- 6,362,620 transactions
- 10 columns of transaction data
- Only 0.129% fraudulent transactions (extreme class imbalance)
- Transaction types include CASH_IN, CASH_OUT, DEBIT, PAYMENT, and TRANSFER

Key attributes in the data:
- Transaction amount
- Origin and destination account balances (before and after)
- Transaction type
- Step (time unit)

The dataset file is included in this repository.

## 🛠️ Approach

### Feature Engineering

I created custom features that significantly improved fraud detection:
- Balance difference calculations
- Transaction amount relative to account balance
- Balance error detection
- Zero balance indicators
- Transaction pattern identification

### Model Selection

After testing multiple algorithms, the Random Forest classifier emerged as the top performer, although XGBoost and CatBoost showed similar excellent results. The model was trained on a carefully balanced dataset with a 60:1 ratio of legitimate to fraudulent transactions.

### Key Finding

One of the most revealing discoveries: **transactions that leave accounts with zero balances** are strongly indicative of fraud. This pattern, along with balance inconsistencies, forms the backbone of the detection system.

## 📈 Results

| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| Random Forest | 0.99993 | 1.00000 | 0.99574 | 0.99787 |
| XGBoost | 0.99991 | 0.99939 | 0.99513 | 0.99756 |
| CatBoost | 0.99992 | 1.00000 | 0.99513 | 0.99757 |
| Logistic Regression | 0.96643 | 0.32592 | 0.98052 | 0.97336 |
| LightGBM | 0.99036 | 0.64509 | 0.91601 | 0.95380 |

Confusion Matrix for Random Forest:
```
[[98556     0]
 [    7  1636]]
```

This means:
- 98,556 legitimate transactions correctly identified
- 1,636 fraudulent transactions caught
- 7 fraudulent transactions missed
- 0 legitimate transactions incorrectly flagged as fraud

## 🔑 Top Fraud Indicators

The most important features for detecting fraud:

1. Balance inconsistencies (20.5%)
2. Original balance changes (17.9%)
3. Transaction amount relative to balance (14.2%)
4. New original balance (11.2%)
5. Zero balance after transaction (7.4%)

## 💻 How to Use This Project

This project is contained in a single Jupyter notebook. You can check it in my Kaggle notebook: https://www.kaggle.com/code/adarshompanigrahi/fraud-detection-accredian-asgnm

### Running the Analysis

You have several options to run or explore this project:

1. **View directly on GitHub**: 
   - The notebook is viewable directly on GitHub with all outputs and visualizations.

2. **Run on Google Colab**:
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/fraud-detection/blob/main/Fraud_Detection.ipynb)
   - Click the badge above to open in Google Colab

3. **Run locally**:
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/fraud-detection.git
   cd fraud-detection
   
   # Create and activate virtual environment (optional)
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   
   # Install the required libraries
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost tensorflow
   
   # Launch Jupyter to run the notebook
   jupyter notebook Fraud_Detection.ipynb
   ```

## 📁 Repository Structure

```
fraud-detection/
├── Fraud_Detection.ipynb  # The complete analysis, EDA, and modeling notebook
├── Fraud.csv              # The dataset file
└── README.md              # Project documentation
```

All code including data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation is contained within the single notebook file.

## 🔮 Future Improvements

- Implement deep learning models for comparison
- Add anomaly detection for novel fraud patterns
- Develop an ensemble approach combining multiple models
- Create a real-time transaction scoring API
- Add interpretability tools to explain model decisions
- Deploy model as a simple API for demonstration purposes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the need for more effective fraud detection systems
- Thanks to the machine learning community for algorithm implementations
