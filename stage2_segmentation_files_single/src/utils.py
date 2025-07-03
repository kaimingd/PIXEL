import torch
import numpy as np
import torch.nn as nn

def score_cal(seg_map,pred_map):
    '''
    labels B * 1
    seg_map B *H * W
    pred_map B * H * W
    '''
    seg_map = np.array(seg_map)
    pred_map = np.array(pred_map)

    total_num = 1
    seg_map = seg_map.reshape(total_num,-1)   
    pred_map = pred_map.reshape(total_num,-1)
    one_hot_map = (pred_map > 128)  
    seg_map = (seg_map > 10)  
    dot_product = (seg_map *one_hot_map).reshape(total_num,-1)  
    
    max_number = np.max(pred_map)
    temp_pred = (pred_map[0] == max_number).astype("int64")   
    flag = int((np.sum(temp_pred * seg_map[0]))>0)
    point_score = flag

    return total_num,point_score

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=False, weight=None):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):  #删去背景
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / (self.n_classes) 


