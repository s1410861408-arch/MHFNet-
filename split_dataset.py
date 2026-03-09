import os
import random
import shutil
from pathlib import Path

def split_dataset(base_path, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    # 設定隨機種子，確保每次劃分的結果一致（方便復現）
    random.seed(seed)
    base_dir = Path(base_path)

    if not base_dir.exists():
        print(f"錯誤：找不到路徑 {base_dir}")
        return

    # 以 images/rgb 資料夾為基準，獲取所有圖片檔名
    rgb_images_dir = base_dir / 'images' / 'rgb'
    if not rgb_images_dir.exists():
        print(f"錯誤：找不到基準圖片資料夾 {rgb_images_dir}")
        return

    # 抓取所有圖片檔案 (支援常見格式)
    image_files = [f for f in os.listdir(rgb_images_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print(f"警告：在 {rgb_images_dir} 中沒有找到圖片！")
        return

    # 提取主檔名 (不含副檔名) 和圖片的副檔名 (例如 .jpg 或 .png)
    file_stems = [Path(f).stem for f in image_files]
    img_ext = Path(image_files[0]).suffix

    # 將檔名列表隨機打亂
    random.shuffle(file_stems)

    # 計算 7:1:2 的切分點
    total_files = len(file_stems)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    splits = {
        'train': file_stems[:train_end],
        'val': file_stems[train_end:val_end],
        'test': file_stems[val_end:]
    }

    # 定義需要同步處理的 4 個子目錄
    sub_dirs = [
        ('images', 'rgb', img_ext),
        ('images', 'sar', img_ext),
        ('labels', 'rgb', '.txt'),
        ('labels', 'sar', '.txt')
    ]

    print(f"開始處理 {base_dir.name}...")
    
    # 建立目標資料夾並移動檔案
    for split_name, stems in splits.items():
        for folder, modality, ext in sub_dirs:
            src_dir = base_dir / folder / modality
            dst_dir = base_dir / folder / modality / split_name
            
            # 確保目標資料夾存在
            dst_dir.mkdir(parents=True, exist_ok=True)

            for stem in stems:
                src_file = src_dir / f"{stem}{ext}"
                dst_file = dst_dir / f"{stem}{ext}"

                # 如果來源檔案存在，就移動到對應的 split 資料夾中
                if src_file.exists():
                    shutil.move(str(src_file), str(dst_file))
                else:
                    print(f"  [警告] 遺失檔案: {src_file}")

    print(f"✅ {base_dir.name} 劃分完成！")
    print(f"   總數: {total_files} -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}\n")

if __name__ == '__main__':
    # 定義你的資料集路徑 (請確認相對於執行此腳本的正確位置)
    qxs_path = './dataset/datasets/QXS-datasets'
    suez_path = './dataset/datasets/Suez-datasets'

    # 執行劃分
    split_dataset(qxs_path)
    split_dataset(suez_path)