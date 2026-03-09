import argparse, sys, os, warnings
warnings.filterwarnings('ignore')

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("D:/YOLO-MIF-master/runs/train2/YOLOv8n-vsff2/weights/best.pt")
    metrics = model.val(data="D:/YOLO-MIF-master/data-vsff2.yaml",
              split='val',
              imgsz=256,
              batch=16,
              channels=4,
              use_simotm='RGBT',
              conf=0.5,
              iou=0.5,
              # rect=False,
              save_json=True,  # if you need to cal coco metrice
              project='runs/val2',
              name='YOLOv8n-vsff2',
              )
    print(metrics.results_dict)
    print(f"Mean Average Precision @.5:.95 : {metrics.box.map}")
    print(f"Mean Average Precision @ .50 : {metrics.box.map50}")
    print(f"Mean Average Precision @ .70 : {metrics.box.map75}")
