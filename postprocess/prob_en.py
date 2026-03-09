from torch.nn import functional as F
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms
import argparse
from os.path import isfile, join
import os
import json
import numpy as np
import cv2
import torch
import pickle
import time
from tqdm import tqdm

def avg_bbox_fusion(match_bbox_vec):
    avg_bboxs = np.sum(match_bbox_vec,axis=0) / len(match_bbox_vec)
    return avg_bboxs

def bayesian_fusion(match_score_vec):
    log_positive_scores = np.log(match_score_vec)
    log_negative_scores = np.log(1 - match_score_vec)
    fused_positive = np.exp(np.sum(log_positive_scores))
    fused_negative = np.exp(np.sum(log_negative_scores))
    fused_positive_normalized = fused_positive / (fused_positive + fused_negative)
    return fused_positive_normalized

def bayesian_fusion_multiclass(match_score_vec, pred_class):
    scores = np.zeros((match_score_vec.shape[0], 4))
    scores[:,:3] = match_score_vec
    scores[:,-1] = 1 - np.sum(match_score_vec, axis=1)
    log_scores = np.log(scores)
    sum_logits = np.sum(log_scores, axis=0)
    exp_logits = np.exp(sum_logits)
    score_norm = exp_logits / np.sum(exp_logits)
    out_score = np.max(score_norm)
    out_class = np.argmax(score_norm)    
    return out_score, out_class


def weighted_box_fusion(bbox, score):
    weight = score / np.sum(score)        
    out_bbox = np.array(bbox) * weight[:,None]
    out_bbox = np.sum(out_bbox, axis=0)    
    return out_bbox

def prepare_data(info1, info2, info3='', method=None):
    out_dict = {}
    for key in info1.keys():
        if key != 'img_name':
            data1 = np.array(info1[key])
            data2 = np.array(info2[key])          
            data_all = np.concatenate((data1, data2), axis=0)
            if info3:
                data3 = np.array(info3[key])
                data_all = np.concatenate((data_all, data3), axis=0)
            out_dict[key] = data_all
    return out_dict


def distance_box_iou(boxes1, boxes2):
    # 简化的DIoU实现
    # 计算IoU
    xx1 = np.maximum(boxes1[:, 0:1], np.transpose(boxes2[:, 0:1]))
    yy1 = np.maximum(boxes1[:, 1:2], np.transpose(boxes2[:, 1:2]))
    xx2 = np.minimum(boxes1[:, 2:3], np.transpose(boxes2[:, 2:3]))
    yy2 = np.minimum(boxes1[:, 3:4], np.transpose(boxes2[:, 3:4]))

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    area1 = (boxes1[:, 2:3] - boxes1[:, 0:1] + 1) * (boxes1[:, 3:4] - boxes1[:, 1:2] + 1)
    area2 = (boxes2[:, 2:3] - boxes2[:, 0:1] + 1) * (boxes2[:, 3:4] - boxes2[:, 1:2] + 1)
    iou = inter / (area1 + area2 - inter)

    # 计算中心点距离
    center1 = (boxes1[:, 0:1] + boxes1[:, 2:3]) / 2  # (N, 1)
    center2 = (boxes2[:, 0:1] + boxes2[:, 2:3]) / 2  # (M, 1)

    # 计算中心点之间的欧氏距离
    center1_expanded = center1[:, np.newaxis, :]  # (N, 1, 1)
    center2_expanded = center2[np.newaxis, :, :]  # (1, M, 1)

    center_dist = np.sqrt(np.sum((center1_expanded - center2_expanded) ** 2, axis=2))  # (N, M)

    # 计算最小闭合矩形的对角线长度
    x1 = np.minimum(boxes1[:, 0:1], np.transpose(boxes2[:, 0:1]))
    y1 = np.minimum(boxes1[:, 1:2], np.transpose(boxes2[:, 1:2]))
    x2 = np.maximum(boxes1[:, 2:3], np.transpose(boxes2[:, 2:3]))
    y2 = np.maximum(boxes1[:, 3:4], np.transpose(boxes2[:, 3:4]))

    diagonal_dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # 计算DIoU
    diou = iou - (center_dist ** 2) / (diagonal_dist ** 2)
    return diou

def nms_bayesian(dict_collect, thresh, method, var=None):
    score_method, box_method = method    
    classes = dict_collect['class']
    dets = dict_collect['bbox']
    scores = dict_collect['score']
    probs = dict_collect['prob']

    # xywh -> xyxy
    x1 = dets[:, 0] - dets[:, 2]/2
    y1 = dets[:, 1] - dets[:, 3]/2
    x2 = dets[:, 0] + dets[:, 2]/2
    y2 = dets[:, 1] + dets[:, 3]/2
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    
    keep = []
    out_classes = []
    match_scores = []
    match_bboxs = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        match = np.where(ovr > thresh)[0]
        match_ind = order[match+1]
        
        match_prob = list(probs[match_ind])
        match_score = list(scores[match_ind])
        match_bbox = list(dets[match_ind][:,:4])
        original_prob = probs[i]
        original_score = scores[i].tolist()
        original_bbox = dets[i][:4]
        
        # If some boxes are matched
        if len(match_score)>0:
            match_score += [original_score]
            match_prob += [original_prob]
            match_bbox += [original_bbox]           
            
            # score fusion
            if score_method == "probEn":
                # final_score, out_class = bayesian_fusion_multiclass(np.asarray(match_prob), classes[i])
                final_score = bayesian_fusion(np.asarray(match_prob))
                # out_classes.append(out_class)          
            elif score_method == 'avg':
                final_score = np.mean(np.asarray(match_score))
                # out_classes.append(classes[i])
            elif score_method == 'max':
                final_score = np.max(match_prob)
                # out_classes.append(classes[i])
            
            # box fusion
            if box_method == "v-avg":
                match_var = list(var[match_ind])                
                original_var = var[i]
                match_var += [original_var]                
                weights = 1/np.array(match_var)
                final_bbox = weighted_box_fusion(match_bbox, np.squeeze(weights))
            elif box_method == 's-avg':
                final_bbox = weighted_box_fusion(match_bbox, match_score)
            elif box_method == 'avg':                
                final_bbox = avg_bbox_fusion(match_bbox)
            elif box_method == 'argmax':                                
                max_score_id = np.argmax(match_score)
                final_bbox = match_bbox[max_score_id]              
            
            match_scores.append(final_score)
            match_bboxs.append(final_bbox)
        else:
            match_scores.append(original_score)
            match_bboxs.append(original_bbox)
            # out_classes.append(classes[i])

        order = order[inds + 1]

        
    assert len(keep)==len(match_scores)
    assert len(keep)==len(match_bboxs)    
    # assert len(keep)==len(out_classes)

    match_bboxs = match_bboxs
    match_scores = torch.Tensor(match_scores)
    # match_classes = torch.Tensor(out_classes)

    # return keep,match_scores,match_bboxs, match_classes
    return keep,match_scores,match_bboxs, None

def fusion(method, info_1, info_2, info_3=''):
    # if method[0] == 'max' and method[1] == 'argmax':
    #     out_boxes, out_scores, out_class = nms_1(info_1, info_2, info_3=info_3)
    # else:
    threshold = 0.4
    dict_collect = prepare_data(info_1, info_2, info3=info_3, method=method)
    keep, out_scores, out_boxes, out_class = nms_bayesian(dict_collect, threshold, method=method)        
    return out_boxes, out_scores, out_class


def preprocess_det(det):
    max_id = int(max([int(det[i]['image_id']) for i in range(len(det))]))
    infos = []
    for i in range(max_id+1):
        info = {}
        info['img_name'] = i
        info['bbox'] = []
        info['score'] = []
        info['class'] = []
        info['prob'] = []
        infos.append(info)
    for i in range(len(det)):
        id = int(det[i]['image_id'])
        infos[id]['bbox'].append(det[i]['bbox'])
        infos[id]['score'].append(det[i]['score'])
        infos[id]['class'].append(det[i]['category_id'])
        infos[id]['prob'].append([det[i]['score']])
    return infos


def apply_late_fusion_and_evaluate(det_1, det_2, method, det_3=''):
    img_folder = '../../../Datasets/FLIR/val/thermal_8_bit/'
    print('Method: ', method)
    start  = time.time()

    jdict = []
    info_1_list = preprocess_det(det_1)
    info_2_list = preprocess_det(det_2)
    if det_3:
        info_3_list = preprocess_det(det_3)
    
    # import pdb; pdb.set_trace()
    
    for i in tqdm(range(len(info_1_list))):
        info_1 = info_1_list[i]
        info_2 = info_2_list[i]
        num_detections = int(len(info_1['bbox']) > 0) + int(len(info_2['bbox']) > 0)
        if det_3:
            info_3 = info_3_list[i]
            num_detections += int(len(info_3['bbox']) > 0)
            
        
        # No detections
        if num_detections == 0:
            continue
        # Only 1 model detection
        elif num_detections == 1:            
            if len(info_1['bbox']) > 0:
                out_boxes = np.array(info_1['bbox'])
                out_class = torch.Tensor(info_1['class'])
                out_scores = torch.Tensor(info_1['score'])
            elif len(info_2['bbox']) > 0:
                out_boxes = np.array(info_2['bbox'])
                out_class = torch.Tensor(info_2['class'])
                out_scores = torch.Tensor(info_2['score'])
            else:
                if det_3:
                    out_boxes = np.array(info_3['bbox'])
                    out_class = torch.Tensor(info_3['class'])
                    out_scores = torch.Tensor(info_3['score'])
        # Only two models with detections
        elif num_detections == 2:
            if not det_3:
                out_boxes, out_scores, out_class = fusion(method, info_1, info_2)
            else:    
                if len(info_1['bbox']) == 0:
                    out_boxes, out_scores, out_class = fusion(method, info_2, info_3)
                elif len(info_2['bbox']) == 0:
                    out_boxes, out_scores, out_class = fusion(method, info_1, info_3)
                else:
                    out_boxes, out_scores, out_class = fusion(method, info_1, info_2)
        # All models detected things
        else:
            out_boxes, out_scores, out_class = fusion(method, info_1, info_2, info_3=info_3)
        # print(info_1)
        # print(info_2)
        # print(out_boxes, out_scores, out_class)
        for j in range(len(out_boxes)):
            jdict.append({'image_id': i, 
                    'category_id': 1.0, 
                    'bbox': [round(box,5) for box in out_boxes[j].tolist()],
                    'score': round(out_scores[j].item(),7)})
    # import pdb; pdb.set_trace()
    
  
            
        

    end = time.time()
    total_time = end - start
    print('Average fps:', 1000/total_time/102)
    #print('Average time:', total_time / int(len(det_2['image'])))
    return jdict       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_path', type=str, default='./runs/val', help='prediction path')
    parser.add_argument('--outfolder', type=str, default='D:/YOLO-MIF-master/runs/val/YOLOv8-allfusion/', help='model.json path')
    parser.add_argument('--score_fusion', type=str, default='max', help='model.json path')
    parser.add_argument('--box_fusion', type=str, default='s-avg', help='model.json path')
    args = parser.parse_args()
    prediction_folder = args.prediction_path
    # dataset = args.dataset_name
    # dataset_folder = args.dataset_path
    out_folder = args.outfolder
    
    # det_file_1 = prediction_folder + '/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSATv1_lframe3_stride3/best_predictions_ct001.json'
    # det_file_2 = prediction_folder + '/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_DSATv1_lframe1_stride1_grad_clip_10/best_predictions_ct001.json'
    # det_file_3 = prediction_folder + '/yolov5l_fusion_transformerx3_Kaist_aligned_tadaconv_mlpSTmixv2_same_num_patches_grad_clip_15/best_predictions_ct001.json'
    det_file_1 = 'D:/YOLO-MIF-master/runs/val/YOLOv8n-RGB/predictions.json'
    det_file_2 = 'D:/YOLO-MIF-master/runs/val/YOLOv8n-RGBT-midfusion-CGA/predictions.json'
    det_file_3 = 'D:/YOLO-MIF-master/runs/val/YOLOv8n-RGBT-midfusion-CGA-shape/predictions.json'
    # val_file_name = 'FLIR_thermal_RGBT_pairs_val.json'
    # val_json_path =  os.path.join(args.dataset_path , val_file_name)
    # val_folder = os.path.join(args.dataset_path , 'thermal_8_bit')

    print('detection file 1:', det_file_1)
    print('detection file 2:', det_file_2)
    print('detection file 3:', det_file_3)
    
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)


    # Read detection results
    det_1 = json.load(open(det_file_1, 'r'))
    det_2 = json.load(open(det_file_2, 'r'))
    det_3 = json.load(open(det_file_3, 'r'))
    
    method = [args.score_fusion, args.box_fusion]
    # 3 inputs
    result = apply_late_fusion_and_evaluate(det_1, det_2, method, det_3=det_3)
    with open(out_folder + '/predictions1.json', 'w') as f:
        json.dump(result, f)
    # 2 inputs only
    #result = apply_late_fusion_and_evaluate(cfg, evaluator, det_2, det_1, method)