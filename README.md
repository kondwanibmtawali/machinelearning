# Machine Learning - MMA Outcome Predictions

This repository contains my completed assignment for a machine learning project in a data science course, focused on predicting outcomes of Mixed Martial Arts (MMA) fights using UFC fighter data.

## Project Overview

The goal of this project is to explore whether relationships between fighter characteristics (e.g., ranking, age, height, wins/losses, fighting style) can be used to predict MMA fight outcomes. The approach involves:

- Merging and preprocessing datasets from Kaggle on UFC fighters and historical fights.
- Simulating hypothetical fights within weight classes to generate a training dataset with derived features (e.g., differences in age, reach, striking/takedown averages).
- Training and tuning machine learning models to predict winners based on simulated data.
- Evaluating the models on real-world historical fight outcomes to assess performance.

**Problem**: Predict MMA fight winners using fighter attributes.  
**Data**: UFC fighter stats and historical fight results from Kaggle.  
**Key Ideas**: Data merging/cleaning, feature engineering (e.g., win percentage, fighting style classification), simulation for training data, model training/evaluation with Decision Trees, Logistic Regression, and K-Nearest Neighbors.  
**Results**: Models achieve ~50% accuracy on test data, with KNN being the most balanced. Challenges include class imbalance and the inherent unpredictability of fights.

**Dataset Sources**:  
- UFC Master Data (fighter rankings, records, etc.) from Kaggle.  
- UFC Fight Data (historical outcomes) from Kaggle.

## Notebook Contents

- **`Machine Learning-MMA Outcomes.ipynb`**: The main Jupyter notebook containing all steps:
  1. Load and merge UFC datasets.
  2. Data cleaning and exploration (fixing columns, adding features like Total Fights, Win Percentage, Fighting Style).
  3. Simulate fights to create training data with derived features (differences between fighters) and probabilistic winners.
  4. Train and tune models (Decision Tree, Logistic Regression, KNN) with hyperparameter tuning and cross-validation.
  5. Preprocess and enrich real-world test data (historical fights).
  6. Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
  7. Analyze results and discuss improvements.

## Technologies Used

- Python
- Jupyter Notebook
- pandas, numpy (data manipulation)
- matplotlib, seaborn (visualization)
- scikit-learn (modeling: DecisionTreeClassifier, LogisticRegression, KNeighborsClassifier, GridSearchCV, classification_report, confusion_matrix)

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mma-outcomes.git  # Replace with your repo URL
