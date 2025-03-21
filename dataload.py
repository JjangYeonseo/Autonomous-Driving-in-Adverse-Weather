import json
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d
from matplotlib.patches import Polygon


# JSON 파일 로딩
def load_json_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


# PCD 파일 로딩
def load_pcd_data(pcd_path):
    return o3d.io.read_point_cloud(pcd_path)


# 다각형 그리기
def draw_polygons_on_image(img, annotations):
    # shapes 항목에서 다각형 정보를 추출
    for shape in annotations.get("shapes", []):
        if shape['shape_type'] == 'polygon':  # 다각형만 그리기
            points = shape['points']
            # Polygon을 그리기 위해 matplotlib의 Polygon 객체 생성
            polygon = Polygon(points, closed=True, edgecolor='r', facecolor='none', linewidth=2)
            plt.gca().add_patch(polygon)


# 이미지와 라벨 시각화 예시
def visualize_data_sample(image_path, json_path, pcd_path=None):
    # 이미지 로딩
    img = Image.open(image_path)
    
    # JSON 데이터 로딩
    annotations = load_json_data(json_path)

    # 이미지를 matplotlib로 출력
    plt.imshow(img)
    plt.title("Image and Annotations")
    
    # 다각형 라벨 시각화
    draw_polygons_on_image(img, annotations)
    
    # 이미지 시각화
    plt.show(block=True)  # 이미지 창이 바로 닫히지 않도록 block=True 추가

    # Lidar 데이터 시각화 (pcd 파일이 있다면)
    if pcd_path:
        pcd = load_pcd_data(pcd_path)
        
        # Open3D 시각화 창을 열고, Lidar 데이터를 시각화
        o3d.visualization.draw_geometries([pcd], window_name="Lidar Data")  # 창 이름 추가


# 경로 설정
image_path = r"path.jpg"
json_path = r"path.json"
pcd_path = r"path.pcd" 
# (path 파일은 형식만 다르고 전부 동일한 소스여야 이미지 매칭 가능)

# 데이터 시각화
visualize_data_sample(image_path, json_path, pcd_path)
