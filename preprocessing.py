import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess_data():

    # Load dataset
    df = pd.read_csv("Dataset/placement.csv")

    # Remove missing values
    df = df.dropna()

    # Convert Yes/No columns into numbers
    categorical_columns = [
        "ExtracurricularActivities",
        "PlacementTraining"
    ]

    for column in categorical_columns:
        df[column] = df[column].map({"Yes": 1, "No": 0})

    # Convert target column
    df["PlacementStatus"] = df["PlacementStatus"].map({
        "Placed": 1,
        "NotPlaced": 0
    })

    # Separate features and target
    X = df.drop("PlacementStatus", axis=1)
    y = df["PlacementStatus"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess_data()

    print("Data preprocessing completed!")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Features:", features)