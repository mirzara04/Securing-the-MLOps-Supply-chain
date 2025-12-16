# src/train.py
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import great_expectations as gx
from security_utils import apply_label_flipping, apply_roni

def run_pipeline(poison_rate=0.1, defense_enabled=False):
    # 1. Setup MLflow and Data Context
    mlflow.set_experiment("Fraud_Detection_Security")
    context = gx.get_context()
    
    with mlflow.start_run():
        # --- DATA STAGE ---
        df = pd.read_csv("data/creditcard.csv").sample(10000) # Small sample for speed
        
        # Defense Layer 1: Data Integrity (Great Expectations)
        # We check the raw data before anything else
        datasource = context.sources.add_pandas(name="my_datasource")
        asset = datasource.add_dataframe_asset(name="my_asset")
        batch = asset.get_batch(dataframe=df)
        
        # (Assuming you created a suite named 'fraud_suite' as in the previous step)
        # validation_result = batch.validate(expectation_suite_name="fraud_suite")
        # mlflow.log_dict(validation_result.to_json_dict(), "ge_validation_results.json")

        X = df.drop('Class', axis=1).values
        y = df['Class'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_val, X_test_clean, y_val, y_test_clean = train_test_split(X_test, y_test, test_size=0.5)

        # --- ATTACK STAGE ---
        y_train_poisoned = apply_label_flipping(y_train, poison_rate)
        mlflow.log_param("poison_rate", poison_rate)

        # --- DEFENSE STAGE ---
        removed_count = 0
        if defense_enabled:
            X_train_final, y_train_final, removed_count = apply_roni(X_train, y_train_poisoned, X_val, y_val)
        else:
            X_train_final, y_train_final = X_train, y_train_poisoned
        
        mlflow.log_param("defense_enabled", defense_enabled)
        mlflow.log_metric("samples_rejected_by_roni", removed_count)

        # --- TRAINING STAGE ---
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train_final, y_train_final)
        
        # --- QUANTITATIVE ANALYSIS ---
        y_pred = clf.predict(X_test_clean)
        auc = roc_auc_score(y_test_clean, clf.predict_proba(X_test_clean)[:, 1])
        
        mlflow.log_metric("auc_score", auc)
        mlflow.sklearn.log_model(clf, "fraud_model")
        
        print(f"Run Complete: Poison={poison_rate}, Defense={defense_enabled}, AUC={auc:.4f}")

if __name__ == "__main__":
    # Example: Run an attack without defense
    run_pipeline(poison_rate=0.2, defense_enabled=False)
    # Example: Run the same attack WITH defense
    run_pipeline(poison_rate=0.2, defense_enabled=True)