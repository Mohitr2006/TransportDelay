# Transport Delay Prediction

This project predicts transport arrival delays using machine learning. It uses information such as weather conditions, traffic, transport type, route details, and scheduled timings to estimate the expected arrival delay.

The project was built to practice the complete machine learning workflow, including data preprocessing, feature engineering, model training, and inference using Scikit-learn.

## Features

- Data cleaning and preprocessing
- Feature engineering from date and time
- Numerical feature scaling using `StandardScaler`
- Categorical feature encoding using `OneHotEncoder`
- Preprocessing with `ColumnTransformer` and `Pipeline`
- Model training using Linear Regression
- Saving and loading models with Joblib
- Prediction on unseen data

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Models Explored

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

After comparing the models, Linear Regression was selected for the final implementation.

## Project Structure

```text
TransportDelayPrediction/
│
├── analysis.ipynb
├── main.py
├── delays.csv
├── model.pkl
├── pipeline.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/Mohitr2006/TransportDelay.git
```

Move into the project directory:

```bash
cd TransportDelay
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

## Future Improvements

- Improve model performance through hyperparameter tuning.
- Experiment with additional machine learning algorithms.
- Build a Flask web application for interactive predictions.
- Train on a larger and more realistic dataset.

## Author

Mohit Raj
