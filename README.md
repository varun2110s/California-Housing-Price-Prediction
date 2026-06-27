🏠 California Housing Price Predictor

Can a machine predict your house price? Apparently yes — and pretty accurately. Built on 20,000+ real California housing records. XGBoost, sklearn pipeline, deployed on Streamlit.

👉 Live App

https://california-housing-price-prediction-ml-model.streamlit.app/


Problem

House prices in California are all over the place. They depend on location, income levels, how close you are to the ocean, and a dozen other things. The goal was to build a model that takes all of this in and spits out an accurate price estimate.

What I Did


Loaded and explored the California Housing dataset (20,000+ records)
Built a full preprocessing pipeline — median imputation for missing values, Standard Scaling for numerical features, One-Hot Encoding for ocean_proximity
Trained 3 models — Linear Regression, Random Forest, and XGBoost
XGBoost gave the best results — lowest RMSE, highest R2 score
Deployed the final model as an interactive web app on Streamlit
