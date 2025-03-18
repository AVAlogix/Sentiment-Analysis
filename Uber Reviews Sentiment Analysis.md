# Uber Reviews Sentiment Analysis 

## Overview
This project analyzes **12,000+ Uber app reviews** from the Google Play Store to extract **sentiment insights** using **NLP techniques** and **machine learning models**. The goal is to classify customer feedback as **Positive, Neutral, or Negative**, and build a predictive model with high accuracy.

## Dataset 
- **Source**: Uber Customer Reviews Dataset on Kaggle  <br>
              https://www.kaggle.com/datasets/kanchana1990/uber-customer-reviews-dataset-2024
- **Columns**:
  - `content`: Review text
  - `score`: Rating (1-5)
  - `sentiment`: Derived label (Positive, Neutral, Negative)
  - Other metadata such as `userName`, `thumbsUpCount`, and `replyContent`

## Project Pipeline 
1. **Data Preprocessing** 
   - Cleaned text (removed stopwords, special characters, and links)
   - Handled missing values
2. **Sentiment Analysis** 
   - Used **VADER** and **TextBlob** to calculate sentiment scores
   - Mapped scores to categorical labels (Positive, Neutral, Negative)
3. **Data Visualization** 
   - Bar plots and word clouds for sentiment distribution
   - Compared VADER and TextBlob sentiment scoring
4. **Machine Learning Models** 
   - Converted text to numerical format using **TF-IDF**
   - Trained **Random Forest** and **XGBoost** classifiers
   - Achieved **Random Forest Accuracy: 89.96%**
   - **Lift: 1.26x over baseline model**
5. **Performance Evaluation** 
   - Compared model accuracy, precision, recall, and F1-score
   - Identified improvement areas for better predictions

## Results 
The sentiment distribution showed ~82% Positive, 9% Neutral, and 9% Negative feedback. The sentiment score distribution plot highlights that both VADER and TextBlob classify most reviews as positive, though their scoring methods differ slightly. Using TF-IDF vectorization, we trained Random Forest and XGBoost classifiers to predict sentiment. Random Forest achieved 89.96% accuracy, outperforming XGBoost (87.00%) with a 1.26x lift over baseline accuracy. These findings suggest strong positive sentiment towards Uber, with opportunities for improvement.

## Notebooks 
- **Jupyter Notebook**: [Sentiment_Analysis.ipynb](#) *(Update with actual link)*

## Next Steps 
- **Hyperparameter tuning** for model improvement
- **Use BERT or LSTMs** for advanced NLP modeling
- **Deploy model as an API** using Flask/FastAPI

## How to Run Locally 
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

