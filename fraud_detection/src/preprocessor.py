from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
TEST_SIZE    = 0.2


def split_data(X, y) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    return (X_train, X_test, y_train, y_test)

def sample_data(X, y, fraction=0.2):
    """Returns a stratified sample of the data for faster training."""
    if fraction >= 1.0:
        return X, y
    _, X_samp, _, y_samp = train_test_split(
        X, y, test_size=fraction, random_state=RANDOM_STATE, stratify=y
    )
    return X_samp, y_samp

def build_pipeline(model, use_smote : bool =True) -> ImbPipeline:
    if use_smote :
        steps = [
            ("scaler", RobustScaler()),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ("model", model)
        ]
    else :
        steps = [
            ("scaler", RobustScaler()),
            ("model", model)
        ]
    
    pipeline = ImbPipeline(steps=steps)
    return pipeline

def get_cv(n_splits :int = 3) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )