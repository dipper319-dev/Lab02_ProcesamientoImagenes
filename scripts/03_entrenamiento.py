import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             confusion_matrix, f1_score,
                             classification_report)
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ── Rutas dinámicas ─────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELOS_DIR = os.path.join(BASE_DIR, "modelos")

# ── Cargar datos ────────────────────────────────────────
X_hog = np.load(os.path.join(MODELOS_DIR, "X_hog.npy"))
X_lbp = np.load(os.path.join(MODELOS_DIR, "X_lbp.npy"))
y     = np.load(os.path.join(MODELOS_DIR, "y.npy"))

clases_nombres = ["Gato", "Perro", "Otro"]

def entrenar_y_evaluar(X, y, nombre_descriptor):
    print(f"\n{'='*50}")
    print(f"  DESCRIPTOR: {nombre_descriptor}")
    print(f"{'='*50}")

    # ── Split 80% train / 20% test ──────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Normalizar ──────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Entrenar SVM ────────────────────────────────────
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm.fit(X_train, y_train)

    # ── Predicciones ────────────────────────────────────
    y_pred = svm.predict(X_test)

    # ── Métricas ────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    f1   = f1_score(y_test, y_pred, average="macro")
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n📊 Métricas:")
    print(f"   Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"   Precisión : {prec:.4f}")
    print(f"   F1-Score  : {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=clases_nombres)}")

    # ── Matriz de confusión ─────────────────────────────
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=clases_nombres,
                yticklabels=clases_nombres)
    plt.title(f"Matriz de Confusión - {nombre_descriptor}")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELOS_DIR, f"confusion_{nombre_descriptor}.png"))
    plt.show()
    print(f"   📁 Matriz guardada en modelos/")

    # ── Guardar modelo y scaler ─────────────────────────
    joblib.dump(svm,    os.path.join(MODELOS_DIR, f"svm_{nombre_descriptor}.pkl"))
    joblib.dump(scaler, os.path.join(MODELOS_DIR, f"scaler_{nombre_descriptor}.pkl"))
    print(f"   💾 Modelo guardado: svm_{nombre_descriptor}.pkl")

    return svm, scaler

# ── Entrenar con HOG ────────────────────────────────────
svm_hog, scaler_hog = entrenar_y_evaluar(X_hog, y, "HOG")

# ── Entrenar con LBP ────────────────────────────────────
svm_lbp, scaler_lbp = entrenar_y_evaluar(X_lbp, y, "LBP")