import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def find_latest_result():
    """outputs/test_ 날짜 폴더 중 가장 최신을 찾아서 반환"""
    output_root = "outputs"
    subdirs = [d for d in os.listdir(output_root) if d.startswith("test_")]
    if not subdirs:
        raise FileNotFoundError("❌ 'outputs/test_xxx' 폴더가 존재하지 않습니다.")
    latest = sorted(subdirs)[-1]
    return os.path.join(output_root, latest, "test_results.csv")

def evaluate(results_path=None, class_path="data/preprocessing/classes.txt"):
    if results_path is None:
        results_path = find_latest_result()

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"❌ 결과 파일이 존재하지 않습니다: {results_path}")

    df = pd.read_csv(results_path)

    # ✅ 컬럼 이름 맞게 수정
    if "true_label" not in df.columns or "pred_label" not in df.columns:
        raise ValueError("❌ 컬럼 이름이 'true_label', 'pred_label' 이어야 합니다.")

    y_true = df['true_label']
    y_pred = df['pred_label']

    with open(class_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines()]

    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Accuracy: {acc:.2f}")

    print("\n📋 Classification Report:")
    used_labels = sorted(set(y_true) | set(y_pred))
    used_class_names = [classes[i] for i in used_labels]
    print(classification_report(y_true, y_pred, target_names=used_class_names, zero_division=0, labels=used_labels))



    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Save to same folder as test result
    result_dir = os.path.dirname(results_path)
    save_path = os.path.join(result_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"✅ Confusion matrix saved to {save_path}")

if __name__ == "__main__":
    evaluate()
