# Used Cars Price Prediction Project

## Project Description
This project aims to predict the prices of used cars based on various features from Craigslist listings. Using machine learning regression techniques, the model analyzes factors such as vehicle year, manufacturer, model, condition, mileage, fuel type, transmission, and more to estimate market prices accurately.

The project utilizes a dataset containing over 400,000 used car listings with 26 features, focusing on building a robust predictive model that can help buyers and sellers make informed decisions in the used car market.

## Technologies Used
- **Python** for data processing and modeling
- **Pandas & NumPy** for data manipulation
- **Scikit-learn** for machine learning pipelines and preprocessing
- **CatBoost, XGBoost, LightGBM** for gradient boosting models
- **Streamlit** for web application deployment
- **Plotly** for data visualizations

## Project Steps

### 1. Data Understanding
- Analyzed the dataset structure with 26 columns including:
  - Basic info: id, url, region, price
  - Vehicle details: year, manufacturer, model, condition, cylinders, fuel, odometer
  - Additional features: title_status, transmission, VIN, drive, size, type, paint_color
  - Location data: lat, long, state, posting_date

### 2. Data Loading
- Imported necessary libraries (pandas, numpy, plotly, sklearn, etc.)
- Loaded the vehicles.csv dataset
- Configured pandas display options

### 3. Data Exploration
- Checked data types and basic information
- Analyzed summary statistics for numerical and categorical columns
- Identified and handled missing values (<5% threshold for dropping)
- Checked for duplicate entries

### 4. Data Cleaning
- Converted 'year' column from float to integer
- Converted 'posting_date' to datetime format
- Removed rows with missing values in critical columns

### 5. Outlier Detection and Removal
- Analyzed price distribution and identified outliers
- Used IQR method to remove extreme price outliers
- Ensured data quality for modeling

### 6. Data Preprocessing for Machine Learning
- Split data into features (X) and target (price, y)
- Created preprocessing pipelines:
  - **Numerical Pipeline**: RobustScaler for scaling
  - **Categorical Pipelines**:
    - OneHotEncoder for transmission and drive
    - BinaryEncoder for high-cardinality features (region, manufacturer, model, etc.)
    - OrdinalEncoder for condition
- Combined all pipelines using ColumnTransformer

### 7. Model Selection and Evaluation
- Tested multiple regression models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - XGBoost Regressor
  - CatBoost Regressor
  - LightGBM Regressor
- Evaluated models with and without target scaling (log transformation)
- Used 5-fold cross-validation with R² scoring
- Selected top 4 performing models: Random Forest, XGBoost, CatBoost, LightGBM

### 8. Hyperparameter Tuning
- Performed RandomizedSearchCV on CatBoost model
- Tuned parameters: depth, learning_rate, iterations
- Achieved optimal hyperparameters for best performance

### 9. Final Model Training and Evaluation
- Trained final CatBoost model on full dataset
- Evaluated performance metrics
- Saved the trained model using joblib

### 10. Deployment
- Created Streamlit web application
- Built interactive price prediction interface
- Deployed model for real-time predictions

## Files Structure
- `Home.py` - Main Streamlit application
- `pages/Price_Prediction.py` - Price prediction page
- `Used Cars Mid Project ML regression.ipynb` - Complete analysis notebook
- `vehicles.csv` - Raw dataset
- `Cleaned_df.parquet` - Processed dataset
- `Catboost_Model.pkl` - Trained model
- `requirements.txt` - Python dependencies
- `catboost_info/` - Model training logs and metadata

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the Streamlit app: `streamlit run Home.py`
3. Access the web interface for price predictions

## Model Performance
The final CatBoost model achieved strong performance in predicting used car prices, with robust cross-validation scores and minimal overfitting after hyperparameter optimization.