import json

def parse_image_label(label_path, class2idx):
    """
    이미지 라벨 파일을 읽고 클래스 인덱스 리스트를 반환합니다.
    """
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        labels = []
        for shape in data.get("shapes", []):
            cls = shape.get("label", "").strip().lower()
            if cls in class2idx:
                labels.append(class2idx[cls])

        return labels  # 하나의 리스트만 반환
    except Exception as e:
        print(f"[경고] 이미지 라벨 파싱 오류: {label_path} → {e}")
        return []

def parse_lidar_label(label_path, class2idx):
    """
    라이다 라벨 파일을 읽고 클래스 인덱스 리스트를 반환합니다.
    """
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        labels = []
        for obj in data.get("objects", []):
            cls = obj.get("classTitle", "").strip().lower()
            if cls in class2idx:
                labels.append(class2idx[cls])

        return labels
    except Exception as e:
        print(f"[경고] 라이다 라벨 파싱 오류: {label_path} → {e}")
        return []
