from torch import nn
import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter
import random
from torch.autograd import Function
from attention import CA_Block


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)

class Bottleneck4(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck4, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup))
    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 128, 2, 2],
    [4, 128, 2, 2],
    [4, 128, 2, 2]]

class MobileFaceNet(nn.Module):
    def __init__(self,
                 num_class, bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFaceNet, self).__init__()
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.blocks1 = Bottleneck4(64,128,2,2)
        self.blocks2 = Bottleneck4(128, 128, 1, 2)

        self.blocks3 = Bottleneck4(128, 128, 2, 4)
        self.blocks4 = Bottleneck4(128, 128, 1, 4)

        self.blocks5 = Bottleneck4(128, 128, 2, 4)
        self.blocks6 = Bottleneck4(128, 128, 1, 4)

        self.conv2 = ConvBlock(128, 512, 1, 1, 0)

        # 20(10), 4(2), 8(4)
        self.linear7 = ConvBlock(512, 512, (8, 20), 1, 0, dw=True, linear=True)
        # self.linear7 = ConvBlock(512, 512, (4, 10), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        self.fc_out = nn.Linear(128, num_class)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, label):
        # x = self.conv1(x) # 64,157
        x = self.dw_conv1(x)
        x = self.blocks1(x)  # 这层的输出(2,128,32,79)
        x = self.blocks2(x)  # 32，79
        x = self.blocks3(x) # 16，40
        x = self.blocks4(x) # （2，128，16，40）
        x = self.blocks5(x) # （2，128，8，20）
        x = self.blocks6(x)  # （2，128，8，20）
        x = self.conv2(x)  # （2，512，8，20）
        x = self.linear7(x) # （2，512，1，1）
        feature512 = x.view(x.size(0), -1) #（2，512）
        x = self.linear1(x)
        feature128 = x.view(x.size(0), -1)
        out = self.fc_out(feature128)
        return out, feature128,feature512



class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(TgramNet, self).__init__()
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        # self.DRSN = BasicBlock(128,128,1)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                # 313(10) , 63(2), 126(4)
                nn.LayerNorm(313),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for idx in range(num_layer)])

    def forward(self, x):
        out = self.conv_extrctor(x)
        out = self.conv_encoder(out)
        return out

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


def mask_rect_logmel(
    x,
    freq_mask_param=16,
    time_mask_param=32,
    num_rect_masks=2,
    mask_value="zero",
    min_freq_mask=1,
    min_time_mask=1):
    if x.dim() != 4:
        raise ValueError(
            f"x must be a 4D tensor [B, C, F, T], "
            f"but got shape {tuple(x.shape)}"
        )

    if num_rect_masks < 0:
        raise ValueError("num_rect_masks must be >= 0.")

    if freq_mask_param <= 0 or time_mask_param <= 0:
        raise ValueError(
            "freq_mask_param and time_mask_param must be > 0."
        )

    B, C, Freq, Time = x.shape

    max_f_len = min(freq_mask_param, Freq)
    max_t_len = min(time_mask_param, Time)

    min_f_len = min(max(min_freq_mask, 1), max_f_len)
    min_t_len = min(max(min_time_mask, 1), max_t_len)

    x_masked = x.clone()
    mask = torch.zeros_like(x)

    for b in range(B):
        if mask_value == "mean":
            fill_value = x[b].mean().detach()
        elif mask_value == "zero":
            fill_value = x.new_tensor(0.0)
        else:
            raise ValueError(
                "mask_value must be 'mean' or 'zero'."
            )

        for _ in range(num_rect_masks):
            f_len = random.randint(min_f_len, max_f_len)
            t_len = random.randint(min_t_len, max_t_len)

            f0 = random.randint(0, Freq - f_len)
            t0 = random.randint(0, Time - t_len)

            f1 = f0 + f_len
            t1 = t0 + t_len

            x_masked[b, :, f0:f1, t0:t1] = fill_value
            mask[b, :, f0:f1, t0:t1] = 1.0

    return x_masked, mask



class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),

            nn.Conv2d(channels,channels,kernel_size=1),
            nn.BatchNorm2d(channels),)

        self.act = nn.PReLU(channels)

    def forward(self, x):
        return self.act(x + self.block(x))

class CrossModalMelReconstruction3(nn.Module):
    def __init__(self, hidden_dim=32, use_residual=True):
        super().__init__()

        self.use_residual = use_residual

        # 初始跨模态融合
        self.input_fusion = nn.Sequential(
            nn.Conv2d(2,hidden_dim, kernel_size=3,padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(hidden_dim))

        # 局部时频建模 + 通道融合
        self.encoder = DepthwiseResidualBlock(hidden_dim)
        self.encoder2 = DepthwiseResidualBlock(hidden_dim)

        self.out = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU())

        self.mel_decoder = nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim,kernel_size=3,padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(hidden_dim),
            nn.Conv2d(hidden_dim,1,kernel_size=1))

    def forward(self, x_mel_masked, x_t):
        x_origin = torch.cat([x_mel_masked, x_t],dim=1)

        h = self.input_fusion(x_origin)
        h = self.encoder(h)
        h = self.encoder2(h)
        out = self.out(h)

        x_mel_rec = self.mel_decoder(h)
        return x_mel_rec, out


class STgramMFN(nn.Module):
    def __init__(self, num_classes,
                 num_domains,
                 c_dim=128,
                 win_len=1024,
                 hop_len=512,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 use_arcface=False, m=0.5, s=30, sub=1):
        super(STgramMFN, self).__init__()

        self.arcface = ArcMarginProduct(in_features=128,out_features=num_classes, m=0.7,s=s,sub=1)

        self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
        self.mobilefacenet = MobileFaceNet(num_class=num_classes,bottleneck_setting=bottleneck_setting)

        self.label_embedding = nn.Embedding(num_embeddings=num_classes,embedding_dim=32)  # 把每个离散类别ID映射成一个长度为32的可学习稠密向量

        # 这个分类器的层很重要，BN  Relu 等有决定性作用
        self.domain_classifie1 = nn.Sequential(
            nn.Linear(128+32, 64),
            # nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_domains))
        self.use_mel_recon = CrossModalMelReconstruction3()
        self.use_mel_recon1 = CrossModalMelReconstruction3()

    def get_tgram(self, x_wav):
        return self.tgramnet(x_wav)

    def forward(self, x_wav, x_mel, label=None,domain_label=None,machine_label=None, lambda_grl=0.0):
        x_mel,x_wav= x_mel.unsqueeze(1),x_wav.unsqueeze(1)
        x_t = self.tgramnet(x_wav).unsqueeze(1)

        mask_config = {"freq_mask_param": 16,"time_mask_param": 32,"num_rect_masks": 2,"mask_value": "zero"}
        x_mel_masked, mel_mask = mask_rect_logmel(x_mel,**mask_config)

        mask_config1 = {"freq_mask_param": 16, "time_mask_param": 32, "num_rect_masks": 2, "mask_value": "zero"}
        x_t_masked,t_mask = mask_rect_logmel(x_t, **mask_config1)

        x_mel_rec, h = self.use_mel_recon(x_mel_masked,x_t)
        x_t_rec, h1 = self.use_mel_recon1(x_mel,x_t_masked)

        x_fused = torch.cat([h,h1], dim=1)

        rec_dict = {"x_mel_target": x_mel,"mel_mask": mel_mask,"x_mel_rec": x_mel_rec,
                    "x_t_target": x_t,    "t_mask": t_mask, "x_t_rec": x_t_rec,}

        fine_out, feature, feature512= self.mobilefacenet(x_fused, label)
        machine_out = None
        if self.arcface is not None and label is not None:
            fine_out = self.arcface(feature, label)

        # 原始  + 位置编码
        rev_feature = grad_reverse(feature, lambda_grl)
        label_emb = self.label_embedding(label)
        # label_emb = self.label_embedding(label).detach()  # 细节
        domain_input = torch.cat([rev_feature, label_emb],dim=1)
        domain_out = self.domain_classifie1(domain_input)
        # domain_out = None
        return fine_out,machine_out, domain_out, feature,feature512,rec_dict


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, sub=1, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub = sub
        self.weight = Parameter(torch.Tensor(out_features * sub, in_features))
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if self.sub > 1:
            cosine = cosine.view(-1, self.out_features, self.sub)
            cosine, _ = torch.max(cosine, dim=2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        # print(x.device, label.device, one_hot.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output



if __name__ == '__main__':


    net = STgramMFN(
        num_classes=10,
        use_arcface=True,
        win_len=1024,
        hop_len=512)

    # 生成合规输入
    x_wav = torch.randn(2, 159744)  # batch_size=2, L=159744
    x_mel = torch.randn(2, 128, 313)  # 梅尔频谱图
    """low(0): 生成随机整数的下限（包含）。即生成的随机整数将从0开始。high(10): 生成随机整数的上限（不包含）。即生成的随机整数会小于
    10，但不包括10。size((2,)): 这是一个元组，指定生成的张量的形状。这里的(2, )表示生成一个包含"""
    labels = torch.randint(0, 10, (2,))  # 假设标签 2 个元素的一维张量。因为输入的是两个样本

    # 前向传播
    logits, features = net(x_wav, x_mel, labels)

    # 打印输出形状
    print("Logits Shape:", logits.shape)  # [2, 10]
    print("Features Shape:", features.shape)  # [2, 128]