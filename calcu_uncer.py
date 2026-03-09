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
            class_id = int(parts[0])  # 类别ID
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
    # 将YOLO格式转换为角点格式
    corners1 = yolo_to_corners(box1, img_width, img_height)
    corners2 = yolo_to_corners(box2, img_width, img_height)

    # 提取坐标
    _, x1_1, y1_1, x2_1, y2_1 = corners1
    _, x1_2, y1_2, x2_2, y2_2 = corners2

    # 计算交集区域的坐标
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # 检查是否没有交集
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算交集区域面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算两个边界框的面积
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # 计算并集区域面积
    union_area = box1_area + box2_area - intersection_area

    # 计算IoU
    iou = intersection_area / union_area
    return iou

def calculate_score(iou):
    """根据IoU计算score"""
    if iou > 0.5:
        return 0.0
    else:
        return 0.5 - iou


def save_score_with_box1(folder1, folder2, output_folder, img_width, img_height):
    """处理两个文件夹中的txt文件，计算IoU并将最大IoU与文件夹1中的边界框信息一起保存"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹1中的所有txt文件
    files1 = [f for f in os.listdir(folder1) if f.endswith('.txt')]

    for filename in files1:
        # 读取文件夹1中的边界框
        file_path1 = os.path.join(folder1, filename)
        bboxes1 = read_bboxes_from_txt(file_path1)

        # 读取文件夹2中的边界框
        file_path2 = os.path.join(folder2, filename)
        bboxes2 = read_bboxes_from_txt(file_path2)

        # 计算最大IoU和相应的score并保存与文件夹1中的边界框信息
        output_lines = []
        for box1 in bboxes1:
            max_iou = 0.0
            for box2 in bboxes2:
                iou = calculate_iou(box1, box2, img_width, img_height)
                if iou > max_iou:
                    max_iou = iou
            # 计算score
            score = calculate_score(max_iou)
            # 保留两位小数，并将score与文件夹1中的边界框信息一起保存
            output_lines.append(f"{score:.2f} {box1[1]} {box1[2]} {box1[3]} {box1[4]}\n")

        # 保存结果到输出文件夹
        output_file_path = os.path.join(output_folder, filename)
        with open(output_file_path, 'w') as file:
            file.writelines(output_lines)

# 示例用法
folder1 = './datasets/labels/rgb/train/'
folder2 = './datasets/labels/sar/train/'
output_folder = './datasets/labels/rgb/train_uncer/'
img_width = 256  # 图像宽度
img_height = 256  # 图像高度
save_score_with_box1(folder1, folder2, output_folder, img_width, img_height)
