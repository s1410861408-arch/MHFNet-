import os

def read_bboxes_from_txt(txt_file_path):
    """从txt文件中读取边界框。假设每行包含一个边界框，格式为class x1 y1 x2 y2"""
    bboxes = []
    with open(txt_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):  # 跳过空行和注释行
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Expected 5 values, got {len(parts)} in line: {line}")
            class_id = int(float(parts[0]))
            #class_id = int(parts[0])   类别ID
            x1, y1, x2, y2 = map(float, parts[1:])
            bboxes.append([class_id, x1, y1, x2, y2])
    return bboxes

def yolo_to_corners(yolo_box, img_width, img_height):
    """将YOLO格式的边界框转换为角点格式"""
    class_id, x_center, y_center, width, height = yolo_box
    x1 = (x_center - width / 2) * img_width
    y1 = (y_center - height / 2) * img_height
    x2 = (x_center + width / 2) * img_width
    y2 = (y_center + height / 2) * img_height
    return [class_id, x1, y1, x2, y2]

def calculate_iou(box1, box2, img_width, img_height):
    """计算两个YOLO格式边界框的IoU（交并比）"""
    corners1 = yolo_to_corners(box1, img_width, img_height)
    corners2 = yolo_to_corners(box2, img_width, img_height)

    _, x1_1, y1_1, x2_1, y2_1 = corners1
    _, x1_2, y1_2, x2_2, y2_2 = corners2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0

def calculate_score(iou):
    """根据IoU计算score (作者原版逻辑)"""
    if iou > 0.5:
        return 0.0
    else:
        return 0.5 - iou

def save_score_with_box1(folder1, folder2, output_folder, img_width, img_height):
    """处理两个文件夹中的txt文件，计算IoU并将最大IoU与文件夹1中的边界框信息一起保存"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files1 = [f for f in os.listdir(folder1) if f.endswith('.txt')]

    for filename in files1:
        file_path1 = os.path.join(folder1, filename)
        bboxes1 = read_bboxes_from_txt(file_path1)

        file_path2 = os.path.join(folder2, filename)
        # 如果另一半模态对应的文件不存在，默认为空列表
        bboxes2 = read_bboxes_from_txt(file_path2) if os.path.exists(file_path2) else []

        output_lines = []
        for box1 in bboxes1:
            max_iou = 0.0
            for box2 in bboxes2:
                iou = calculate_iou(box1, box2, img_width, img_height)
                if iou > max_iou:
                    max_iou = iou
            
            score = calculate_score(max_iou)
            # 保留两位小数，并将score与文件夹1中的边界框信息一起保存
            output_lines.append(f"{score:.2f} {box1[1]} {box1[2]} {box1[3]} {box1[4]}\n")

        output_file_path = os.path.join(output_folder, filename)
        with open(output_file_path, 'w') as file:
            file.writelines(output_lines)

def process_all_datasets():
    img_width = 256
    img_height = 256
    splits = ['train', 'val', 'test']

    # 1. 处理 QXS 数据集 (以 RGB 为基准)
    print("🚀 开始处理 QXS 数据集 (基准: RGB)...")
    qxs_base = './dataset/datasets/QXS-datasets/labels/'
    for split in splits:
        folder1 = os.path.join(qxs_base, f'rgb/{split}/')
        folder2 = os.path.join(qxs_base, f'sar/{split}/')
        output_folder = os.path.join(qxs_base, f'rgb/{split}_uncer/')
        
        if os.path.exists(folder1) and os.path.exists(folder2):
            save_score_with_box1(folder1, folder2, output_folder, img_width, img_height)
            print(f"  ✅ 已生成 QXS {split} 的不确定性标签 -> {output_folder}")
        else:
            print(f"  ❌ [警告] 找不到 QXS {split} 的标签文件夹，请检查上一步的划分是否成功。")

    # 2. 处理 Suez 数据集 (以 SAR 为基准)
    print("\n🚀 开始处理 Suez 数据集 (基准: SAR)...")
    suez_base = './dataset/datasets/Suez-datasets/labels/'
    for split in splits:
        folder1 = os.path.join(suez_base, f'sar/{split}/')
        folder2 = os.path.join(suez_base, f'rgb/{split}/')
        output_folder = os.path.join(suez_base, f'sar/{split}_uncer/')
        
        if os.path.exists(folder1) and os.path.exists(folder2):
            save_score_with_box1(folder1, folder2, output_folder, img_width, img_height)
            print(f"  ✅ 已生成 Suez {split} 的不确定性标签 -> {output_folder}")
        else:
            print(f"  ❌ [警告] 找不到 Suez {split} 的标签文件夹。")

if __name__ == '__main__':
    process_all_datasets()
    print("\n🎉 所有不确定性标签 (Uncertainty Labels) 生成完毕！")