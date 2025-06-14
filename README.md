# credit-card-fraud-detection-system
# Credit Card Fraud Detection System

A machine learning-based web application for detecting fraudulent credit card transactions using multiple classification models.

## Overview

This project implements a real-time credit card fraud detection system using machine learning. It compares the performance of three different models:
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Logistic Regression

The system provides an interactive web interface where users can input transaction details and get instant fraud predictions along with detailed model performance metrics.

## Features

- **Multiple Model Comparison**: Compare predictions from SVM, KNN, and Logistic Regression models
- **Interactive Web Interface**: User-friendly Streamlit-based interface
- **Real-time Predictions**: Instant fraud detection for new transactions
- **Comprehensive Metrics**: View detailed performance metrics including:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC AUC
- **Visual Analytics**: 
  - Model performance comparison charts
  - ROC curves
  - Confusion matrices

## Prerequisites

- Python 3.7+
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the web application:
```bash
streamlit run web_app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the sidebar to input transaction details:
   - Adjust the sliders for each feature
   - Click "Check for Fraud" to get predictions

4. View the results:
   - Individual model predictions
   - Performance metrics comparison
   - Visual analytics

## Data

The system uses the Credit Card Fraud Detection dataset from Kaggle. For the demo version, it uses a sample of 10,000 transactions to ensure quick response times.

## Model Details

### SVM (Support Vector Machine)
- Uses probability estimates for predictions
- Effective for high-dimensional data
- Good at finding complex decision boundaries

### KNN (K-Nearest Neighbors)
- Uses 5 nearest neighbors for classification
- Non-parametric approach
- Simple but effective for fraud detection

### Logistic Regression
- Linear model with probability estimates
- Fast and interpretable
- Good baseline model for comparison

## Performance Considerations

- The demo version retrains models each time for demonstration purposes
- For production use, it's recommended to:
  - Use saved, pre-trained models
  - Implement proper model versioning
  - Add model monitoring and retraining pipelines

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Credit Card Fraud Detection dataset from Kaggle
- Streamlit for the web interface
- scikit-learn for machine learning models
