
# DMart E-commerce Machine Learning Project

## Project Overview
This project applies machine learning to e-commerce data containing
25,000 records. It demonstrates order-value prediction, experimental
order-cancellation classification, and model performance evaluation.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Streamlit
- Jupyter Notebook

## Machine Learning Tasks

### 1. Regression
Predict customer order value using customer, product and shopping
behaviour features.

Best tested R² score: 0.5603

Models:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Polynomial Regression
- Random Forest
- XGBoost

### 2. Classification
Experimental prediction of possible order cancellation.

Models:
- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- XGBoost

Classification performance was weak, with ROC-AUC values close
to 0.50. Predictions should not be used for business decisions.

## Streamlit Application
The application includes:
- Project Overview
- Order Value Prediction
- Cancellation Prediction
- Model Performance Dashboard

## Project Structure
- Dataset/
- Notebooks/
- Models/
- Visualizations/
- Reports/
- Streamlit_App/

## How to Run
Install the required libraries:

    pip install -r requirements.txt

Launch the application:

    cd Streamlit_App
    python -m streamlit run app.py

## Important Limitations
The regression model explains approximately 56% of the variation
in the test data. Cancellation classification is experimental
and requires further improvement.

## Author
Data Analytics and Machine Learning Portfolio Project
