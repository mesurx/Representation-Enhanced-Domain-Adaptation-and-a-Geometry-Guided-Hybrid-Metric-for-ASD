import torch
import torch.nn as nn
import torch.nn.functional as F


class ASDLoss(nn.Module):
    def __init__(self):
        super(ASDLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        loss = self.ce(logits, labels)
        return loss


"20260613 改进，在对“同类别但不同 domain”的样本对赋予更高权重，从而鼓励模型学习更加 domain-invariant 的类别特征的基础上，只计算只用当前 batch 中确实存在的同类样本作为正样本来计算损失"
class DomainAwareSupConLoss2(nn.Module):
    def __init__(self, temperature=0.07, cross_domain_weight=2.0):
        super(DomainAwareSupConLoss2, self).__init__()
        self.temperature = temperature
        self.cross_domain_weight = cross_domain_weight

    def forward(self, features, labels, domains):
        """
        features: [batch_size, feature_dim]
        labels:   [batch_size]
        domains:  [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]

        # 建议归一化，否则 matmul 是普通点积，不是 cosine similarity
        # features = F.normalize(features, dim=1)

        labels = labels.contiguous().view(-1, 1)
        domains = domains.contiguous().view(-1, 1)

        # [B, B]，同类别为 1
        same_class = torch.eq(labels, labels.T).float().to(device)

        # [B, B]，不同 domain 为 1
        diff_domain = torch.ne(domains, domains.T).float().to(device)

        # 去掉自己和自己的对比
        logits_mask = torch.ones_like(same_class) - torch.eye(batch_size, device=device)

        # 正样本：同类别且不是自己
        pos_mask = same_class * logits_mask

        # 跨域同类正样本权重更大
        weight = torch.ones_like(pos_mask)
        weight = weight + diff_domain * (self.cross_domain_weight - 1.0)
        weight = weight * pos_mask

        # 相似度矩阵
        logits = torch.matmul(features, features.T) / self.temperature

        # 数值稳定
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 分母排除自身
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log( exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # 每个 anchor 的正样本权重和
        weight_sum = weight.sum(dim=1)

        # 只对正样本位置计算加权平均 log probability
        mean_log_prob_pos = (weight * log_prob).sum(dim=1) / (weight_sum + 1e-12)

        # 只保留有正样本的 anchor
        valid_anchor = weight_sum > 0

        # 如果整个 batch 都没有正样本，返回 0 loss
        if valid_anchor.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -mean_log_prob_pos[valid_anchor].mean()

        return loss

"这个损失就是前期很快达到最优"
class HierarchicalSupConLossV2(nn.Module):
    def __init__(
        self,
        temperature=0.2,
        strong_weight=1.2,
        weak_weight=1,
        min_valid_anchors=4,
        normalize=True):

        super(HierarchicalSupConLossV2, self).__init__()
        self.temperature = temperature
        self.strong_weight = strong_weight
        self.weak_weight = weak_weight
        self.min_valid_anchors = min_valid_anchors
        self.normalize = normalize

    def forward(self, features, labels, machine_labels):
        device = features.device
        batch_size = features.size(0)

        if batch_size <= 1:
            return torch.zeros([], device=device, requires_grad=True)

        if self.normalize:
            features = F.normalize(features, dim=1)

        labels = labels.contiguous().view(-1, 1)
        machine_labels = machine_labels.contiguous().view(-1, 1)

        same_label = torch.eq(labels, labels.T).float().to(device)   # 两两样本同label矩阵
        same_machine = torch.eq(machine_labels, machine_labels.T).float().to(device)   # 两两样本同机器矩阵

        logits_mask = torch.ones_like(same_label) - torch.eye(batch_size, device=device)  # 减去对角线

        strong_pos_mask = same_label * logits_mask   # 同机器同标签
        weak_pos_mask = same_machine * (1.0 - same_label) * logits_mask  # 同机器不同标签

        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)   # 相似度越大值越小，log的值越大

        strong_count = strong_pos_mask.sum(dim=1)  # 统计强样本数量
        weak_count = weak_pos_mask.sum(dim=1)    # 统计弱样本数量

        valid_anchor_mask = (strong_count + weak_count) > 0

        if valid_anchor_mask.sum() < self.min_valid_anchors:   # 至少4个正样本或弱样本
            return torch.zeros([], device=device, requires_grad=True)

        strong_log_prob = (strong_pos_mask * log_prob).sum(dim=1) / (strong_count + 1e-12)  # 之所以求均值，因为可能batch正样本的数量可能不同
        weak_log_prob = (weak_pos_mask * log_prob).sum(dim=1) / (weak_count + 1e-12)

        # 没有 strong 或 weak 的 anchor，对应项置 0
        strong_valid = strong_count > 0
        weak_valid = weak_count > 0

        per_anchor_loss = torch.zeros(batch_size, device=device) # 初始化

        per_anchor_loss[strong_valid] += (
            -self.strong_weight * strong_log_prob[strong_valid])

        per_anchor_loss[weak_valid] += (
            -self.weak_weight * weak_log_prob[weak_valid])


        # 对每个 anchor 的有效分支权重做归一化 # 有些样本仅有强正样本，但有些样本呢既有强，也有弱正样本
        anchor_weight = torch.zeros(batch_size, device=device)
        anchor_weight[strong_valid] += self.strong_weight
        anchor_weight[weak_valid] += self.weak_weight

        per_anchor_loss = per_anchor_loss / (anchor_weight + 1e-12)

        loss = per_anchor_loss[valid_anchor_mask].mean()

        return loss