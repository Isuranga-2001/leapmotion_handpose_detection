import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train_random_forest(csv_path, model_out="model/rf/rf_leapmotion_gestures.pkl", test_size=0.2, n_estimators=200, random_state=42):
    """
    Train a Random Forest classifier on LeapMotion gesture dataset.

    Args:
        csv_path: Path to CSV dataset with LeapMotion features + 'class_name' column.
        model_out: Path to save trained model.
        test_size: Proportion of dataset used for test split.
        n_estimators: Number of trees in Random Forest.
        random_state: Random seed for reproducibility.
    """
    print(f"[INFO] Loading dataset from {csv_path}")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("[WARNING] Dataset contains missing values. Dropping rows with NaN values.")
        df = df.dropna()
    
    # Prepare features and target
    # Exclude non-feature columns: timestamp, hand_type, class_name
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'hand_type', 'class_name']]
    X = df[feature_cols].values
    y = df['class_name'].values
    
    print(f"[INFO] Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"[INFO] Classes found: {np.unique(y)}")
    print(f"[INFO] Class distribution:")
    for class_name in np.unique(y):
        count = np.sum(y == class_name)
        print(f"  {class_name}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Encode labels if they are strings
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=random_state
    )
    
    print(f"[INFO] Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    
    # Train Random Forest model
    print(f"[INFO] Training Random Forest with {n_estimators} trees...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate on test set
    print("[INFO] Evaluating model...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n[RESULT] Test Accuracy: {acc*100:.2f}%\n")
    
    # Detailed classification report
    class_names = label_encoder.classes_
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:20s}: {row['importance']:.4f}")
    
    # Save model and label encoder
    model_data = {
        'model': clf,
        'label_encoder': label_encoder,
        'feature_columns': feature_cols,
        'test_accuracy': acc
    }
    
    joblib.dump(model_data, model_out)
    print(f"\n[INFO] Model saved to {model_out}")
    print(f"[INFO] Model includes: classifier, label encoder, and feature column names")
    
    return clf, label_encoder, feature_cols, acc


def load_model(model_path):
    """
    Load trained model and associated data.
    
    Returns:
        model: Trained RandomForestClassifier
        label_encoder: LabelEncoder for class names
        feature_columns: List of feature column names
        test_accuracy: Test accuracy of the model
    """
    model_data = joblib.load(model_path)
    return (
        model_data['model'], 
        model_data['label_encoder'], 
        model_data['feature_columns'], 
        model_data['test_accuracy']
    )


def predict_gesture(model, label_encoder, feature_columns, feature_vector):
    """
    Predict gesture from feature vector.
    
    Args:
        model: Trained RandomForestClassifier
        label_encoder: Fitted LabelEncoder
        feature_columns: List of expected feature column names
        feature_vector: numpy array of features (same order as feature_columns)
    
    Returns:
        predicted_class: Predicted gesture class name
        confidence: Prediction confidence (probability of predicted class)
    """
    # Ensure feature_vector is 2D
    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)
    
    # Predict
    prediction = model.predict(feature_vector)[0]
    probabilities = model.predict_proba(feature_vector)[0]
    
    # Decode prediction
    predicted_class = label_encoder.inverse_transform([prediction])[0]
    confidence = np.max(probabilities)
    
    return predicted_class, confidence


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Random Forest for LeapMotion hand gesture classification")
    parser.add_argument("--csv", type=str, default="data/rf_data/data_bottom.csv", 
                       help="Path to input CSV dataset (default: ../../data/rf_data/data_bottom.csv)")
    parser.add_argument("--model_out", type=str, default="model/rf/rf_leapmotion_gestures.pkl", 
                       help="Path to save trained model (default: model/rf/rf_leapmotion_gestures.pkl)")
    parser.add_argument("--test_size", type=float, default=0.2, 
                       help="Test size fraction (default: 0.2)")
    parser.add_argument("--n_estimators", type=int, default=200,
                       help="Number of trees in Random Forest (default: 200)")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV file not found: {args.csv}")
        exit(1)
    
    # Train the model
    train_random_forest(
        csv_path=args.csv,
        model_out=args.model_out,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        random_state=args.random_state
    )
