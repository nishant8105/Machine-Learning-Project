import optuna
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from src.preprocessor import build_pipeline, get_cv

RF  = "Random Forest"
LR  = "Logistic Regression"
GBC = "Gradient Boosting"
RANDOM_STATE =42

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", 'models')

# Add at top of file after imports
optuna.logging.set_verbosity(optuna.logging.WARNING)

def compare_models(X, y, use_smote : bool = True) -> dict :
    models = {
        LR : LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        RF: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE),
        GBC : GradientBoostingClassifier(random_state=RANDOM_STATE)
    }
    results = {}
    for name, model in models.items():
        pipeline = build_pipeline(model=model, use_smote=use_smote)
        scores = cross_val_score(
            pipeline,
            X,
            y,
            scoring='f1',
            cv = get_cv(),
            n_jobs=-1
            )
        
        results[name] = {
            'mean_f1' : np.mean(scores),
            'std_f1' : np.std(scores)
        }
    return results

def get_best_model(results : dict):
    best_model = max(results, key=lambda name : results[name]['mean_f1'])
    return best_model


def tune_model(model_name, X, y, use_smote=True, n_trials=50):
    
    def objective(trial):
        if model_name == RF:
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300),
                max_depth=trial.suggest_int("max_depth", 3, 15),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
        
        elif model_name == LR:
            model = LogisticRegression(
                C=trial.suggest_float("C", 0.001, 100, log=True),
                solver=trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        
        elif model_name == GBC:
            model = GradientBoostingClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                random_state=RANDOM_STATE
            )
        
        else:
            raise ValueError("Invalid model_name")
        
        pipeline = build_pipeline(model, use_smote=use_smote)
        
        scores = cross_val_score(
            pipeline,
            X,
            y,
            scoring="f1",
            cv=get_cv()
        )
        
        return scores.mean()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_value, study.best_params, model_name


def train_best_model(model_name, best_params, X, y, use_smote=True):
    
    if model_name == RF:
        model = RandomForestClassifier(**best_params, n_jobs=-1, random_state=RANDOM_STATE)
    
    elif model_name == LR:
        model = LogisticRegression(
            **best_params,
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    
    elif model_name == GBC:
        model = GradientBoostingClassifier(**best_params, random_state=RANDOM_STATE)
    
    else:
        raise ValueError("Invalid model_name")
    
    pipeline = build_pipeline(model, use_smote=use_smote)
    pipeline.fit(X, y)
    
    return pipeline

def save_model(pipeline, model_name : str) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    save_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    joblib.dump(pipeline, save_path)
    
    return save_path