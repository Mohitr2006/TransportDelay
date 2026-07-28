import os

import joblib
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attributes, cat_attributes):
    #for numerical colunms
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    #for categorical columns
    cat_pipelines = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    #Construct Full Pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attributes),
        ("cat", cat_pipelines, cat_attributes)
    ])

    return full_pipeline

def preprocess(df):

    df["event_type"] = df["event_type"].fillna("No Event")

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
    df["hour"] = df["time"].dt.hour
    df["minute"] = df["time"].dt.minute

    df["scheduled_departure"] = pd.to_datetime(
        df["scheduled_departure"],
        format="%H:%M:%S"
    )
    df["departure_hour"] = df["scheduled_departure"].dt.hour
    df["departure_minute"] = df["scheduled_departure"].dt.minute

    df["scheduled_arrival"] = pd.to_datetime(
        df["scheduled_arrival"],
        format="%H:%M:%S"
    )
    df["arrival_hour"] = df["scheduled_arrival"].dt.hour
    df["arrival_minute"] = df["scheduled_arrival"].dt.minute

    
    df["arrival_cat"] = pd.cut(
            df["actual_arrival_delay_min"],
            bins=[-4, 0, 5, 13, 21, 30],
            labels=[1, 2, 3, 4, 5]
        )
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(df, df['arrival_cat']):
        x_train = df.loc[train_index]
        x_test = df.loc[test_index]

    y_train = x_train[["actual_arrival_delay_min"]]
    y_test = x_test[["actual_arrival_delay_min"]]

    #Removing the arrival_cat and not required columns
    drop_attribs = [
        "trip_id",
        "date",
        "time",
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure_delay_min",
        "actual_arrival_delay_min",
        "delayed",
        "arrival_cat"
    ]
    for dataset in (x_train, x_test):
        dataset.drop(drop_attribs, axis=1, inplace=True)

    return x_train, x_test, y_train, y_test


if not os.path.exists(MODEL_FILE):

    df = pd.read_csv("delays.csv")

    x_train, x_test, y_train, y_test = preprocess(df)

    df = x_train.copy()

    num_attributes = [
        "temperature_C",
        "humidity_percent",
        "wind_speed_kmh",
        "precipitation_mm",
        "event_attendance_est",
        "traffic_congestion_index",
        "holiday",
        "peak_hour",
        "weekday",
        "month",
        "hour",
        "minute",
        "departure_hour",
        "departure_minute",
        "arrival_hour",
        "arrival_minute"
    ]

    cat_attributes = df.drop(num_attributes, axis=1).columns.tolist()

    pipeline = build_pipeline(num_attributes, cat_attributes)

    delay_prepared = pipeline.fit_transform(df)
    print(delay_prepared)

    model = LinearRegression()
    model.fit(delay_prepared, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is Trained!")

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    df = pd.read_csv("delays.csv")

    x_train, x_test, y_train, y_test = preprocess(df)

    x_verify = x_test.copy()
    x_verify['actual_arrival_delay_min'] = y_test
    x_verify.to_csv("verify.csv", index=False)

    transformed_input = pipeline.transform(x_test)
    predictions = model.predict(transformed_input)
    x_test['actual_arrival_delay_min'] = predictions

    x_test.to_csv("output.csv", index=False)
    print("Intereference Completed")
