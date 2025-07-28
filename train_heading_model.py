import json
import numpy as np
import random
from collections import defaultdict
# from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
import joblib

def train_and_save_model(data_path="jsondata/bestdata2.json", max_body=10000, model_path="best_model.pkl"):
    with open(data_path, encoding='utf8') as f:
        data = json.load(f)

    grouped_by_label = defaultdict(list)
    for item in data:
        grouped_by_label[item["level"]].append(item)

    samples = []
    for label, items in grouped_by_label.items():
        if label == "body":
            sampled = random.sample(items, max_body)
        else:
            sampled = items
        samples.extend(sampled)

    random.shuffle(samples)

    # Extract features
    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [item["text"] for item in samples]
    sizes = [item["size"] for item in samples]
    bboxes = [item["bbox"] for item in samples]
    fonts = [item["font"] for item in samples]
    labels = [item["level"] for item in samples]

    # text_embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    # Font one-hot encoding
    font_enc = OneHotEncoder(handle_unknown='ignore')
    font_features = font_enc.fit_transform(np.array(fonts).reshape(-1, 1)).toarray()

    # Layout features: size + bbox + font
    layout_raw = np.hstack([np.array(sizes).reshape(-1, 1), np.array(bboxes), font_features])

    # Normalize layout features
    layout_scaler = StandardScaler()
    layout_features = layout_scaler.fit_transform(layout_raw)

    # Final feature vector
    X = np.hstack([layout_features])

    # Encode labels
    le = LabelEncoder()
    le.fit(["H1", "H2", "H3", "H4", "H5", "title", "body"])
    y = le.transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=40)

    model_scores = {}

    # MLP
    print("\n🔹 Training: MLP (RandomizedSearchCV)")
    mlp_param_grid = {
        'hidden_layer_sizes': [(128,), (256,), (128, 64), (256, 128)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
    }
    mlp_base = MLPClassifier(max_iter=750)
    mlp_best = RandomizedSearchCV(mlp_base, mlp_param_grid, n_iter=20, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    mlp_best.fit(X_train, y_train)
    mlp_score = mlp_best.score(X_test, y_test)
    model_scores['MLP'] = (mlp_score, mlp_best)
    print("\n🔸 MLP Classification Report")
    print(classification_report(y_test, mlp_best.predict(X_test), labels=le.transform(le.classes_), target_names=le.classes_))

    # XGBoost
    print("\n🔹 Training: XGBoost")

    xgb_best = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
    xgb_best.fit(X_train, y_train)
    xgb_score = xgb_best.score(X_test, y_test)
    model_scores['XGBoost'] = (xgb_score, xgb_best)
    print("\n🔸 XGBoost Classification Report")
    print(classification_report(y_test, xgb_best.predict(X_test), labels=le.transform(le.classes_), target_names=le.classes_))

    # LightGBM
    print("\n🔹 Training: LightGBM (RandomizedSearchCV)")
    lgbm_param_grid = {
        'n_estimators': [50, 100, 150],
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [-1, 5, 10],
        'subsample': [0.7, 0.8, 1.0]
    }
    lgbm_base = LGBMClassifier(verbose=-1)
    lgbm_best = RandomizedSearchCV(lgbm_base, lgbm_param_grid, n_iter=15, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    lgbm_best.fit(X_train, y_train)
    lgbm_score = lgbm_best.score(X_test, y_test)
    model_scores['LightGBM'] = (lgbm_score, lgbm_best)
    print("\n🔸 LightGBM Classification Report")
    print(classification_report(y_test, lgbm_best.predict(X_test), labels=le.transform(le.classes_), target_names=le.classes_))

    # Select best model
    best_model_name = max(model_scores, key=lambda k: model_scores[k][0])
    best_score, best_model = model_scores[best_model_name]
    print(f"\n✅ Best model: {best_model_name} with accuracy: {best_score:.4f}")

    # Save best model and encoders
    joblib.dump(best_model, model_path)
    # joblib.dump(model, "sentence_transformer.pkl")
    joblib.dump(font_enc, "font_encoder.pkl")
    joblib.dump(layout_scaler, "layout_scaler.pkl")
    joblib.dump(le, "label_encoder.pkl")

    print(f"\n✅ Model saved as '{model_path}'")

if __name__ == "__main__":
    train_and_save_model()
