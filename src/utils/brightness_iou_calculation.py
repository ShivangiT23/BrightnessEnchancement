from scipy.optimize import linear_sum_assignment
import numpy as np

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def get_pairwise_IOU(preds,gt_bb):
        iou_list = []
        for i,pred in enumerate(preds):
                temp=[]
                for j,gt in enumerate(gt_bb):
                        iou = bb_intersection_over_union(pred,gt)
                        temp.append(iou)
                iou_list.append(temp)
        return iou_list

def one_image_IOU(pred_bboxes,gt_bboxes,iou_thresh=0.4):
    # pred_bbox -> x1,y1,x2,y2,score,label
    #print("IOU pred:" ,pred_bboxes[:2])
    #print("IOU gt:" ,gt_bboxes[:2])
    fp_pred = []
    tp_pred = []
    tp_gt = []
    fn_gt = []

    if(not len(pred_bboxes)):
        print("All {} are not detected".format(len(gt_bboxes)))
        fn_gt.extend(gt_bboxes)
    elif(not len(gt_bboxes)):
        print("No Actual detected")
        fp_pred.extend(pred_bboxes)
    else:
        cost_matrix=get_pairwise_IOU(pred_bboxes,gt_bboxes)
        modified_cost_matrix=np.array(cost_matrix)

        row_ind,col_ind=linear_sum_assignment(modified_cost_matrix,maximize=True)

        for i in range(len(row_ind)):
            iou = cost_matrix[row_ind[i]][col_ind[i]]
            if(iou > iou_thresh):
                tp_pred.append(pred_bboxes[row_ind[i]])
                tp_gt.append(gt_bboxes[col_ind[i]])

            else:
                fp_pred.append(pred_bboxes[row_ind[i]])

        fn_gt = [i for i in gt_bboxes if i not in tp_gt]

        fp_pred.extend([i for i in pred_bboxes if i not in fp_pred and i not in tp_pred])

    if len(fp_pred)==0 and len(fn_gt)==0:
        return "correct"
    # print(len(gt_bboxes),len(pred_bboxes))
    # print(len(fp_pred),len(fn_gt))
    return "wrong"
