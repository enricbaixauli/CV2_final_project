import torch
import torch.nn as nn

class TokenGuideLoss(nn.Module):
    def __init__(self, eps=1e-7,alpha=0.6, region_num=9):
        """
        overall_loss = alpha*pred_loss + (1-alpha)*ori_loss
        """
        super().__init__()
        assert region_num == 9 or region_num == 11, "Region number should be 9 or 11."
        self.eps = eps
        self.alpha = alpha
        self.region_num = region_num


    def Identity(self, cont_labels):
        """
        Identity function in ori_loss
        """
        # cont_labels: [batchsize, (pitch_agl, yaw_agl, roll_agl)]
        pitch = cont_labels[0]
        yaw = cont_labels[1]
        flag = -1
        if self.region_num == 9:
            if yaw > 40 and pitch > 30:
                flag = 0
            # 1
            elif yaw >= -40 and yaw <= 40 and pitch > 30:
                flag = 1
            # 2
            elif yaw < -40 and pitch > 30:
                flag = 2
            # 3
            elif yaw > 40 and pitch >= -30 and pitch <= 30:
                flag = 3
            # 4
            elif yaw >= -40 and yaw <= 40 and pitch >= -30 and pitch <= 30:
                flag = 4
            # 5
            elif yaw < -40 and pitch >= -30 and pitch <= 30:
                flag = 5
            # 6
            elif yaw > 40 and pitch < -30:
                flag = 6
            # 7
            elif yaw >= -40 and yaw <= 40 and pitch < -30:
                flag = 7
            # 8
            elif yaw < -40 and pitch < -30:
                flag = 8

        elif self.region_num == 11:
            # 0
            if yaw > 40 and pitch > 30:
                flag = 0
            # 1
            elif yaw >= -40 and yaw <= 40 and pitch > 30:
                flag = 1
            # 2
            elif yaw < -40 and pitch > 30:
                flag = 2
            # 3
            elif yaw > 60 and pitch >= -30 and pitch <= 30:
                flag = 3
            # 4
            elif yaw > 20 and yaw <= 60 and pitch >= -30 and pitch <= 30:
                flag = 4
            # 5
            elif yaw >= -20 and yaw <= 20 and pitch >= -30 and pitch <= 30:
                flag = 5
            # 6
            elif yaw >= -60 and yaw < -20 and pitch >= -30 and pitch <= 30:
                flag = 6
            # 7
            elif  yaw < -60 and pitch >= -30 and pitch <= 30:
                flag = 7
            # 8
            elif yaw > 40 and pitch < -30:
                flag = 8
            # 9
            elif yaw >= -40 and yaw <= 40 and pitch < -30:
                flag = 9
            # 10
            elif yaw < -40 and pitch < -30:
                flag = 10
        return flag


    def G_loss(self,m1, m2):
        """
        GeodesicLoss
        """
        # both matrices are orthogonal rotation matrices
        m = torch.bmm(m1, m2.transpose(1, 2))  # shape: batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2

        theta = torch.acos(torch.clamp(cos, -1 + self.eps, 1 - self.eps))

        return torch.mean(theta)


    def forward(self, m1, m2, cont_labels, dir_9_d):
        """
        m1: ground truth rotation matrix.
        m2: predicted rotation matrix.
        cont_labels: gt Euler angles in each orientation.
        dir_9_d: predicted rotation matrix of each region.
        """
        # identify the correct region
        batch_size = m1.shape[0]
        rgn_pred = torch.zeros_like(m2) # region prediction
        for batch in range(batch_size):
            # in each batch
            location = self.Identity(cont_labels[batch,...])
            all_regions = dir_9_d[batch, ...]
            rgn_pred[batch,...] = all_regions[location,...]

        pred_loss = self.G_loss(m1,m2)
        ori_loss = self.G_loss(m1,rgn_pred)
        overall_loss = self.alpha*pred_loss+(1-self.alpha)*ori_loss
        return overall_loss, pred_loss, ori_loss
