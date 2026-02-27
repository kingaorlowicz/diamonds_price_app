# 💎 Diamond Price Estimator
An interactive Streamlit application for predicting diamond prices based on their physical and quality characteristics. The prediction engine is powered by an XGBoost regression model, trained in my separate machine learning project.
## Project Overview
his application allows users to input key diamond attributes and instantly receive a price estimate. It is designed as a simple, intuitive interface for exploring how different features influence diamond pricing.
The model takes into account:
- carat
- depth
- table
- cut
- color
- clarity

The app processes the input, encodes categorical variables, aligns them with the model’s expected feature set, and returns a predicted price.
## App
You can try the app here: https://diamondspriceapp.streamlit.app/

## Machine Learning Model
The price prediction is generated using an XGBoost Regressor.
This model was trained as part of my separate project focused on diamond price prediction, including:
- Exploratory data analysis
- Preprocessing
- Hyperparameter tuning
- Model comparison and evaluation
You can find the full training and model development here: https://github.com/kingaorlowicz/diamonds_ml_project

## Technologies Used
- Python
- Streamlit
- Pandas
- XGBoost
- Joblib

## Project Structure
```
├── app.py
├── xgboost_diamonds_model.pkl
├── model_columns.pkl
├── requirements.txt
└── README.md
```

