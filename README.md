# 🏦 Universal Bank – Personal Loan Predictor Dashboard

An interactive Streamlit dashboard that analyzes and predicts personal loan acceptance using **7 machine learning models** on the Universal Bank dataset.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 📊 Pages

| Page | Description |
|---|---|
| 📊 Overview & EDA | KPIs, distributions, correlations |
| 🤖 Model Performance | Metrics table, confusion matrices, ranking |
| 📈 ROC & Thresholds | ROC curves, CV AUC, precision-recall explorer |
| 🔍 Feature Analysis | Feature importances, scatter plots, violin plots |
| 🔮 Predict a Customer | Real-time prediction with gauge visualization |

## 🤖 Models Used

- Logistic Regression
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- Random Forest ⭐
- Gradient Boosting ⭐
- SVM

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/universal-bank-dashboard
cd universal-bank-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud

1. Fork or push this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select this repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** 🎉

## 📁 Project Structure

```
universal-bank-dashboard/
├── app.py                # Main Streamlit application
├── UniversalBank.csv     # Dataset
├── requirements.txt      # Python dependencies
└── README.md
```

## 📦 Dataset

The Universal Bank dataset contains **5,000 customer records** with features:

| Feature | Description |
|---|---|
| Age | Customer's age in years |
| Experience | Years of professional experience |
| Income | Annual income ($000) |
| Family | Family size |
| CCAvg | Avg. monthly credit card spending ($000) |
| Education | 1=Undergrad, 2=Graduate, 3=Advanced |
| Mortgage | Home mortgage value ($000) |
| Securities Account | Has securities account? |
| CD Account | Has CD account? |
| Online | Uses online banking? |
| CreditCard | Has UniversalBank credit card? |
| **Personal Loan** | **Target: Accepted loan offer?** |
