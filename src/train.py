# src/train.py
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import great_expectations as gx
from security_utils import apply_label_flipping, apply_roni
import yaml  # Added for params loading

def run_pipeline():
    # --- 1. LOAD PARAMS FROM SUPPLY CHAIN CONFIG ---
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    # Map YAML values to variables
    poison_rate = params['attack']['poison_rate']
    defense_enabled = params['defense']['enabled']
    roni_threshold = params['defense']['threshold']
    sample_size = params['data']['sample_size']
    n_estimators = params['model']['n_estimators']

    # 2. Setup MLflow and Data Context
    mlflow.set_experiment("Fraud_Detection_Security")
    context = gx.get_context()
    
    with mlflow.start_run():
        # --- 3. DATA STAGE ---
        # Using sample_size from params.yaml
        df = pd.read_csv("data/creditcard.csv").sample(sample_size) 
        
        # Defense Layer 1: Data Integrity (Great Expectations)
        datasource = context.sources.add_pandas(name="my_datasource")
        asset = datasource.add_dataframe_asset(name="my_asset")
        batch = asset.get_batch(dataframe=df)
        
        # (Optional: Validation logic here)

        X = df.drop('Class', axis=1).values
        y = df['Class'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_val, X_test_clean, y_val, y_test_clean = train_test_split(X_test, y_test, test_size=0.5)

        # --- 4. ATTACK STAGE ---
        # Using poison_rate from params.yaml
        y_train_poisoned = apply_label_flipping(y_train, poison_rate)

        # --- 5. DEFENSE STAGE ---
        removed_count = 0
        if defense_enabled:
            # Using threshold from params.yaml
            X_train_final, y_train_final, removed_count = apply_roni(
                X_train, y_train_poisoned, X_val, y_val, threshold=roni_threshold
            )
        else:
            X_train_final, y_train_final = X_train, y_train_poisoned
        
        # --- 6. LOGGING (Audit Trail) ---
        # We log the entire params dictionary so we have a record of the supply chain state
        mlflow.log_params(params['attack'])
        mlflow.log_params(params['defense'])
        mlflow.log_metric("samples_rejected_by_roni", removed_count)

        # --- 7. TRAINING STAGE ---
        # Using n_estimators from params.yaml
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X_train_final, y_train_final)
        
        # --- 8. QUANTITATIVE ANALYSIS ---
        y_pred = clf.predict(X_test_clean)
        auc = roc_auc_score(y_test_clean, clf.predict_proba(X_test_clean)[:, 1])
        
        mlflow.log_metric("auc_score", auc)
        mlflow.sklearn.log_model(clf, "fraud_model")
        
        print(f"✅ Run Complete: Poison={poison_rate}, Defense={defense_enabled}, AUC={auc:.4f}")

if __name__ == "__main__":
    # Now we just call the function. It knows where to find its instructions (params.yaml).
    run_pipeline()