import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import load_and_preprocess_data


# Load preprocessed data
X_train, X_test, y_train, y_test, scaler, features = load_and_preprocess_data()


# Define three ML algorithms
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


results = {}
best_accuracy = 0
best_model = None
best_model_name = ""


# Train and evaluate models
for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print("Accuracy:", round(accuracy * 100, 2), "%")

    print(classification_report(
        y_test,
        predictions,
        target_names=["Not Placed", "Placed"]
    ))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name



# Save all three trained models
for name, model in models.items():

    if name == "Logistic Regression":
        filename = "logistic_regression.pkl"

    elif name == "Decision Tree":
        filename = "decision_tree.pkl"

    else:
        filename = "random_forest.pkl"

    joblib.dump({
        "model": model,
        "scaler": scaler,
        "features": features,
        "model_name": name
    }, filename)

    print(filename, "saved successfully!")


print("\n-----------------------------")
print("MODEL COMPARISON")
print("-----------------------------")

for name, accuracy in results.items():
    print(name, ":", round(accuracy * 100, 2), "%")

print("\nBest Model:", best_model_name)
print("Best Accuracy:", round(best_accuracy * 100, 2), "%")
print("\nModel saved successfully!")

joblib.dump(results, "model_comparison.pkl")