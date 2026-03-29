import os.path

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
DEEPVOICE_DIR = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/May25_rerun_tmp_data"


def evaluate_with_cv(features_file, class1='good', class2='technical_issues', n_splits=5, balance=False):
    with open(features_file, 'rb') as f:
        feature_data = pickle.load(f)

    df = pd.DataFrame(feature_data)
    print("Initial class counts:", df['quality'].value_counts())
    df = df[df['quality'].isin([class1, class2])].reset_index(drop=True)
    print(f"\nSelected classes counts:\n{df['quality'].value_counts()}")

    if balance:
        min_class_size = df['quality'].value_counts().min()
        balanced_df = pd.DataFrame()
        for class_name in [class1, class2]:
            class_data = df[df['quality'] == class_name]
            if len(class_data) > min_class_size:
                class_data = class_data.sample(n=min_class_size, random_state=42)
            balanced_df = pd.concat([balanced_df, class_data])
        df = balanced_df.reset_index(drop=True)
        print(f"\nBalanced classes counts:\n{df['quality'].value_counts()}")

    X = df.drop(['filename', 'quality'], axis=1)
    y = df['quality'].map({class1: 0, class2: 1})
    filenames = df['filename']

    cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
    clf = RandomForestClassifier(n_estimators=1000, random_state=42)

    all_predictions = []
    all_true_values = []
    all_probabilities = []
    all_filenames = []
    aucs = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        clf.fit(X_train, y_train)
        probabilities = clf.predict_proba(X_val)[:, 1]
        predictions = (probabilities > 0.5).astype(int)

        all_predictions.extend(predictions)
        all_true_values.extend(y_val)
        all_probabilities.extend(probabilities)
        all_filenames.extend(filenames.iloc[val_idx])

        fpr, tpr, _ = roc_curve(y_val, probabilities)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)
        print(f"Fold {fold} AUC: {fold_auc:.3f}")

    cm = confusion_matrix(all_true_values, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[class1, class2],
                yticklabels=[class1, class2])
    plt.title(f'Confusion Matrix: {class1} vs {class2} {"(Balanced)" if balance else ""}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'./figures/confusion_matrix_{class1}_vs_{class2}.png')
    plt.close()

    fpr, tpr, _ = roc_curve(all_true_values, all_probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {class1} vs {class2} {"(Balanced)" if balance else ""}')
    plt.legend(loc="lower right")
    plt.savefig(f'./figures/roc_curve_{class1}_vs_{class2}.png')
    plt.close()

    results_df = pd.DataFrame({
        'filename': all_filenames,
        'true_label': [class1 if v == 0 else class2 for v in all_true_values],
        'predicted_label': [class1 if v == 0 else class2 for v in all_predictions],
        'probability': all_probabilities
    })
    results_df.to_csv(os.path.join(DEEPVOICE_DIR, f'CV_evaluation_predictions_{class1}_vs_{class2}.csv'), index=False)

    print(f"\nMean AUC: {np.mean(aucs):.3f} (+/- {np.std(aucs) * 2:.3f})")
    return cm, aucs


def train_and_predict_all(features_file, class1='good', class2='technical_issues', balance=False):
    """
    Train classifier on all labeled data and predict classes for all recordings.

    Args:
        features_file (str): Path to the pickle file containing audio features
        class1 (str): Name of the first class (default: 'good')
        class2 (str): Name of the second class (default: 'technical_issues')
        balance (bool): Whether to balance classes in training data (default: False)

    Returns:
        pd.DataFrame: DataFrame containing predictions for all recordings
    """
    # Load all feature data
    with open(features_file, 'rb') as f:
        feature_data = pickle.load(f)

    full_df = pd.DataFrame(feature_data)

    # Split into labeled and unlabeled data
    labeled_df = full_df[full_df['quality'].isin([class1, class2])].reset_index(drop=True)
    unlabeled_df = full_df[~full_df['quality'].isin([class1, class2])].reset_index(drop=True)

    print("Training data class distribution:")
    print(labeled_df['quality'].value_counts())

    # Balance training data if requested
    if balance:
        min_class_size = labeled_df['quality'].value_counts().min()
        balanced_df = pd.DataFrame()
        for class_name in [class1, class2]:
            class_data = labeled_df[labeled_df['quality'] == class_name]
            if len(class_data) > min_class_size:
                class_data = class_data.sample(n=min_class_size, random_state=42)
            balanced_df = pd.concat([balanced_df, class_data])
        labeled_df = balanced_df.reset_index(drop=True)
        print("\nBalanced training data class distribution:")
        print(labeled_df['quality'].value_counts())

    # Prepare training data
    X_train = labeled_df.drop(['filename', 'quality'], axis=1)
    y_train = labeled_df['quality'].map({class1: 0, class2: 1})

    # Train classifier on all labeled data
    clf = RandomForestClassifier(n_estimators=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Prepare all data for prediction
    X_all = full_df.drop(['filename', 'quality'], axis=1)
    probabilities = clf.predict_proba(X_all)[:, 1]
    predictions = (probabilities > 0.5).astype(int)

    # Create results DataFrame
    results_df = pd.DataFrame({
        'filename': full_df['filename'],
        'original_label': full_df['quality'],
        'predicted_label': [class1 if v == 0 else class2 for v in predictions],
        'probability': probabilities
    })

    # Save predictions to CSV
    results_df.to_csv(os.path.join(DEEPVOICE_DIR, f'all_predictions_{class1}_vs_{class2}.csv'), index=False)

    # Print summary statistics
    print("\nPrediction Summary:")
    print(f"Total recordings analyzed: {len(results_df)}")
    print(f"Predicted class distribution:")
    print(results_df['predicted_label'].value_counts())

    return results_df


if __name__ == "__main__":
    FEATURES_FILE = os.path.join(DEEPVOICE_DIR, "audio_features_for_noise_filtering.pkl")

    # Run cross-validation evaluation
    confusion_matrix, auc_scores = evaluate_with_cv(FEATURES_FILE, 'good', 'technical_issues', balance=False)

    # Train on all labeled data and predict for all recordings
    all_predictions = train_and_predict_all(FEATURES_FILE, 'good', 'technical_issues', balance=False)