import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,LabelEncoder, StandardScaler,TargetEncoder,MinMaxScaler


def create_preprocessing_pipeline():

    StandardScaler_columns = [
        "BMI",
        "Exercise_Time",
        "CheckUp_Year",
        "CheckUp_Month",
        "CheckUp_Day",
        "CheckUp_DayOfWeek"
    ]
    minmax_scaler_columns = [
            "Patient_Age",
            "Salary"
    ]

    onehot_columns = [
        "Hospital_ID",
        "Any_Existing_Disease"]

    Ordinal_column = ["Alcohol"]
    

    StandardScaler_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    minmax_scaler_pipeline = Pipeline ([
            ("imputer", SimpleImputer(strategy="mean")),
            ("minmaxscaler",MinMaxScaler())
    ])

    onehot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    ordinal_pipeline = Pipeline([  
         ("imputer", SimpleImputer(strategy="most_frequent")), 
        ("encoder",OrdinalEncoder())
    ])

    preprocessing_pipeline = ColumnTransformer([
        ("standard", StandardScaler_pipeline, StandardScaler_columns),
        ("MinMax", minmax_scaler_pipeline, minmax_scaler_columns),
        ("onehot", onehot_pipeline, onehot_columns),
        ("label",ordinal_pipeline,Ordinal_column)
    ])

    return preprocessing_pipeline