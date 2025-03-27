import os
import json
import shutil
from tqdm import tqdm
import pandas as pd
import random
from collections import defaultdict, Counter

# 사용자 설정 경로
LABEL_ROOT = r"C:/Users/dadab/Desktop/project data/traindata/labellingdata"
DATA_ROOT = r"C:/Users/dadab/Desktop/project data/traindata/sourcedata"

# 저장 경로
OUTPUT_DIR = "data/preprocessing"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 설정
MAX_SAMPLES_PER_CLASS = 2000  # 각 클래스당 최대 샘플 수 (균형 잡기 위함)

# 라벨링 구분 함수
def is_lidar_label(label_path):
    return label_path.endswith(".pcd.json")

def is_valid_image_label(label_data):
    return "shapes" in label_data and len(label_data["shapes"]) > 0

def is_valid_lidar_label(label_data):
    return "objects" in label_data and len(label_data["objects"]) > 0

def parse_label_file(label_path):
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if is_lidar_label(label_path):
            return "lidar", data if is_valid_lidar_label(data) else None
        else:
            return "image", data if is_valid_image_label(data) else None
    except:
        return None, None

# 유효 데이터 수집
class_to_samples = defaultdict(list)
class_set = set()

for weather_folder in tqdm(os.listdir(LABEL_ROOT), desc="날씨 조건별 폴더 탐색"):
    weather_path = os.path.join(LABEL_ROOT, weather_folder)
    if not os.path.isdir(weather_path):
        continue

    for condition_folder in os.listdir(weather_path):
        condition_path = os.path.join(weather_path, condition_folder)
        for time_folder in os.listdir(condition_path):
            time_path = os.path.join(condition_path, time_folder)

            for direction in os.listdir(time_path):
                direction_path = os.path.join(time_path, direction)
                if not os.path.isdir(direction_path):
                    continue

                files = os.listdir(direction_path)
                if len(files) == 0:
                    shutil.rmtree(direction_path)
                    continue

                for file in files:
                    if not file.endswith(".json"):
                        continue

                    label_path = os.path.join(direction_path, file)
                    label_type, label_data = parse_label_file(label_path)

                    if label_data is None:
                        continue

                    base_filename = file.replace(".pcd.json", "").replace(".json", "")
                    modality = "lidar" if label_type == "lidar" else "image"

                    weather_key = weather_folder.replace("TL_", "TS_")
                    src_base = os.path.join(DATA_ROOT, weather_key, condition_folder, time_folder, direction)

                    data_ext = ".pcd" if modality == "lidar" else ".jpg"
                    data_path = os.path.join(src_base, base_filename + data_ext)
                    if not os.path.exists(data_path):
                        continue

                    # 클래스 추출 (가장 대표적인 하나)
                    if modality == "image":
                        labels = [shape["label"].strip().lower() for shape in label_data["shapes"]]
                    else:
                        labels = [obj["classTitle"].strip().lower() for obj in label_data["objects"]]

                    if not labels:
                        continue

                    dominant_class = Counter(labels).most_common(1)[0][0]
                    class_set.update(labels)

                    sample_info = {
                        "weather": condition_folder,
                        "time": time_folder,
                        "direction": direction,
                        "filename": base_filename,
                        "modality": modality,
                        "data_path": data_path,
                        "label_path": label_path,
                        "dominant_class": dominant_class
                    }

                    class_to_samples[dominant_class].append(sample_info)

# 균형 잡힌 샘플링
balanced_samples = []
for cls, samples in class_to_samples.items():
    if len(samples) > MAX_SAMPLES_PER_CLASS:
        balanced_samples.extend(random.sample(samples, MAX_SAMPLES_PER_CLASS))
    else:
        balanced_samples.extend(samples)

# 저장
df = pd.DataFrame(balanced_samples)
df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w", encoding='utf-8') as f:
    for cls in sorted(class_set):
        f.write(cls + "\n")

print(f"[완료] 클래스 수: {len(class_set)}개")
print(f"[완료] 저장된 균형 샘플 수: {len(balanced_samples)}개")
