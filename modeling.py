import wandb
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import metrics
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
from config import MODEL_SAVE_PATH  # Ensure this import for the save path

numerical_features = [
    'Sum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300', 'TypeOfSteel_A400',
    'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index',
    'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index',
    'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index',
    'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas'
]

def build_pipeline():
    """
    Build a machine learning pipeline that includes preprocessing, feature selection, and model training.
    """
    numerical_transformer = Pipeline(steps=[
        ('scaler', RobustScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_classif, k='all')),
        ('model', XGBClassifier(learning_rate=0.01, n_estimators=300, objective='binary:logistic'))
    ])
    
    return pipeline

def train_and_evaluate(X_train, y_train, X_test, y_test, target_name):
    """
    Train the model and evaluate its performance, logging metrics to W&B.
    """
    pipeline = build_pipeline()
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    
    wandb.log({
        f"{target_name} Accuracy": accuracy,
        f"{target_name} F1 Score": f1,
        f"{target_name} Precision": precision,
        f"{target_name} Recall": recall
    })
    
    print(f"Target: {target_name}")
    print("Accuracy: ", accuracy)
    print("F1 Score: ", f1)
    print("Precision: ", precision)
    print("Recall: ", recall)
    
def save_model(model):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, MODEL_SAVE_PATH)
