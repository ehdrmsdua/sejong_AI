import os
import torch
from glob import glob
from PIL import Image
import numpy as np
from torchvision import transforms
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
import segmentation_models_pytorch as smp
import argparse
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

def save_object_detection_results(pred, save_path):
    # 이미지 크기를 모델 입력 크기로 설정
    model_input_width, model_input_height = 352, 640
    result_lines = []
    for i, det in enumerate(pred):
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                x_center = ((xyxy[0] + xyxy[2]) / 2) / model_input_width
                y_center = ((xyxy[1] + xyxy[3]) / 2) / model_input_height
                width = (xyxy[2] - xyxy[0]) / model_input_width
                height = (xyxy[3] - xyxy[1]) / model_input_height
                result_line = f"0 {x_center.item()} {y_center.item()} {width.item()} {height.item()}"
                result_lines.append(result_line)

    with open(save_path, 'w') as f:
        f.write('\n'.join(result_lines))

def predict_images_in_folder(image_folder, object_detection_model_path, segmentation_model_path, device, save_dir):
    object_detection_model = attempt_load(object_detection_model_path)  # 객체 탐지 모델 로드
    object_detection_model.to(device).eval()

    segmentation_model = smp.DeepLabV3Plus(  # 도로 영역 분류 모델 로드
        encoder_name="resnet101",
        encoder_weights=None,
        in_channels=3,
        classes=4
    )
    segmentation_model.load_state_dict(torch.load(segmentation_model_path, map_location=device))
    segmentation_model.to(device).eval()

    transform = transforms.Compose([  # 이미지 전처리
        transforms.Resize((640, 352)),
        transforms.ToTensor(),
    ])

    # 결과 저장 폴더 생성
    detection_save_dir = os.path.join(save_dir, 'detection_results')
    road_area_mask_save_dir = os.path.join(save_dir, 'road_area_masks')
    os.makedirs(detection_save_dir, exist_ok=True)
    os.makedirs(road_area_mask_save_dir, exist_ok=True)

    for image_path in glob(os.path.join(image_folder, '*.jpg')):
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size
        img_tensor = transform(image).unsqueeze(0).to(device)

        pred = object_detection_model(img_tensor, augment=False)[0]
        pred = non_max_suppression(pred, 0.4, 0.6, classes=None, agnostic=False)

        # 객체 위치 라벨링 파일 저장
        object_detection_save_path = os.path.join(detection_save_dir, os.path.basename(image_path).replace('.jpg', '_detection.txt'))
        save_object_detection_results(pred, object_detection_save_path)

        seg_pred = segmentation_model(img_tensor)  # 도로 영역 분류 수행
        seg_pred = torch.argmax(seg_pred, dim=1).byte().cpu().numpy()

        # 도로 영역 분류 라벨링 파일 저장
        road_area_mask = Image.fromarray(seg_pred[0])
        road_area_mask_save_path = os.path.join(road_area_mask_save_dir, os.path.basename(image_path).replace('.jpg', '_road_area_mask.png'))
        road_area_mask.save(road_area_mask_save_path)


## main start
parser = argparse.ArgumentParser(description='교통 CCTV 돌발상황 분석 프로그램')

# 입력받을 인자 
parser.add_argument('--data_dir', required=True, help='이미지 경로를 입력하세요')
parser.add_argument('--result_dir', required=True, help='분석결과가 저장될 경로를 입력하세요')

# 인자값 저장
args = parser.parse_args()
print(args)

# 이미지 폴더에서 파일 경로 불러오기
file_pathes = glob(os.path.join(args.data_dir,'*.jpg'))
print(file_pathes)

# 분석결과 저장 폴더 생성
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_folder = args.data_dir  # 분석할 이미지 폴더 경로
object_detection_model_path = r'C:\sejong_test\model\final.pt'  # 객체 탐지 모델 파일 경로
segmentation_model_path = r'C:\sejong_test\model\model_fold4_epoch33.pth'  # 도로 영역 분류 모델 파일 경로
save_dir = args.result_dir  # 결과를 저장할 디렉토리 경로

predict_images_in_folder(image_folder, object_detection_model_path, segmentation_model_path, device, save_dir)



## 두 라벨링 파일 받아서 최종 라벨링 파일 생성

# 새로 예측한 라벨링 파일의 각 폴더 경로 지정
object_predict_label_folder = os.path.join(args.result_dir, 'detection_results')
segmentation_label_folder = os.path.join(args.result_dir, 'road_area_masks')

import numpy as np

def calculate_overlap(road_image_path, human_detection_path, output_folder):
    road_label = np.array(Image.open(road_image_path))
    crosswalk_mask = (road_label == 2)  # 횡단보도 영역
    road_mask = (road_label == 1) # 도로 영역 

    with open(human_detection_path, 'r') as f:
        detections = f.readlines()

    labels = []  # 각 객체의 라벨을 저장할 리스트

    for detection in detections:
        label, x_center, y_center, width, height = [float(x) for x in detection.strip().split()]
        img_width, img_height = road_label.shape[1], road_label.shape[0]

        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)

        human_mask = np.zeros_like(road_label, dtype=bool)
        human_mask[y1:y2, x1:x2] = True

        overlap = np.logical_and(crosswalk_mask, human_mask)
        overlap_ratio = np.sum(overlap) / np.sum(human_mask) if np.sum(human_mask) > 0 else 0

        # 도로와 겹침 비율 
        overlap_road = np.logical_and(road_mask, human_mask)
        overlap_road_ratio = np.sum(overlap_road) / np.sum(human_mask) if np.sum(human_mask) > 0 else 0

        # 횡단보도 영역과의 겹침 비율에 따라 라벨 조정
        if overlap_ratio > 0:
            label = 1  # 횡단보도와 겹치고 도로 영역에 있는 경우
        elif overlap_road_ratio >= 0.2:  # 도로 영역과 20% 이상 겹치는 경우
            label = 0  # 횡단보도와 겹치지 않는 경우
        else:
            continue  # 1, 2에 모두 해당하지 않는 경우 무시

        labels.append([label, x_center, y_center, width, height])

    # 결과를 텍스트 파일로 저장
    output_filename = os.path.basename(human_detection_path)
    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, 'w') as f:
        for label in labels:
            f.write(' '.join(map(str, label)) + '\n')

    print(f"Labels saved to {output_path}")

def process_folders(road_label_folder, human_label_folder, output_folder):
    road_images = sorted(glob(os.path.join(road_label_folder, '*.png')))
    human_detections = sorted(glob(os.path.join(human_label_folder, '*.txt')))

    if len(road_images) != len(human_detections):
        print("The number of files in each folder does not match.")
        return

    for road_image_path, human_detection_path in zip(road_images, human_detections):
        calculate_overlap(road_image_path, human_detection_path, output_folder)

# Example usage
road_label_folder = segmentation_label_folder
human_label_folder = object_predict_label_folder
output_folder = args.result_dir

process_folders(road_label_folder, human_label_folder, output_folder)


## 모든 예측 알고리즘 종료 후 중간 파일 삭제
import shutil

# 코드 실행 후에 폴더 삭제
shutil.rmtree(object_predict_label_folder)
shutil.rmtree(segmentation_label_folder)