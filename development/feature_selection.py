def select_features(df):

    columns_to_drop = [
        "Patient_ID",
        "Contact_Number",
        "Cancer_Detected"
    ]

    X = df.drop(columns_to_drop, axis=1)

    if "Cancer_Detected" in df.columns:    ##user not able to see ...it will predict by model

        y = df["Cancer_Detected"].map({    
            "No": 0,
            "Yes": 1
        })

        return X, y

    return X