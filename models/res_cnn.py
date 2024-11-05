-import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.net_sphere import *
from models.att import *
from models.temporal.tem1 import *
from models.camf import *
from models.sd_cross_atten import CrossAttention, FeedForward
from models.MutualGuidedCoAttention import *
debug = False

'''''
class ResCBAMLayer(nn.Module):
    def __init__(self, in_planes, feature_size):
        super(ResCBAMLayer, self).__init__()
        self.in_planes = in_planes
        self.feature_size = feature_size
        self.ch_AvgPool = nn.AvgPool3d(feature_size, feature_size)
        self.ch_MaxPool = nn.MaxPool3d(feature_size, feature_size)
        self.ch_Linear1 = nn.Linear(in_planes, in_planes // 4, bias=False)
        self.ch_Linear2 = nn.Linear(in_planes // 4, in_planes, bias=False)
        self.ch_Softmax = nn.Softmax(1)
        self.sp_Conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sp_Softmax = nn.Softmax(1)
        self.sp_sigmoid = nn.Sigmoid()
    def forward(self, x):
        #print('x:',x.shape)
        x_ch_avg_pool = self.ch_AvgPool(x).view(x.size(0), -1)
        x_ch_max_pool = self.ch_MaxPool(x).view(x.size(0), -1)
        #print('x_ch_avg_pool:',x_ch_avg_pool.shape)
        #print('x_ch_max_pool:',x_ch_max_pool.shape)
        # x_ch_avg_linear = self.ch_Linear2(self.ch_Linear1(x_ch_avg_pool))
        a = self.ch_Linear1(x_ch_avg_pool)
        #print('a:',a.shape)
        x_ch_avg_linear = self.ch_Linear2(a)
        #print('x_ch_avg_linear:',x_ch_avg_linear.shape)

        x_ch_max_linear = self.ch_Linear2(self.ch_Linear1(x_ch_max_pool))
        #print('x_ch_max_linear:',x_ch_max_linear.shape)
        ch_out = (self.ch_Softmax(x_ch_avg_linear + x_ch_max_linear).view(x.size(0), self.in_planes, 1, 1, 1)) * x
        #print('ch_out:',ch_out.shape)
        x_sp_max_pool = torch.max(ch_out, 1, keepdim=True)[0]
        #print('x_sp_max_pool:',x_sp_max_pool.shape)
        x_sp_avg_pool = torch.sum(ch_out, 1, keepdim=True) / self.in_planes
        #print('x_sp_avg_pool:',x_sp_avg_pool.shape)
        sp_conv1 = torch.cat([x_sp_max_pool, x_sp_avg_pool], dim=1)
        #print('sp_conv1:',sp_conv1.shape)
        sp_out = self.sp_Conv(sp_conv1)
        #print('sp_out:',sp_out.shape)
        sp_out = self.sp_sigmoid(sp_out.view(x.size(0), -1)).view(x.size(0), 1, x.size(2), x.size(3), x.size(4))
        #print('sp_out:',sp_out.shape)
        out = sp_out * x + x
        #print('out:',out.shape)
        return out
'''''

class ResCBAMLayer(nn.Module):
    def __init__(self, in_planes, feature_size):
        super(ResCBAMLayer, self).__init__()
        self.in_planes = in_planes
        self.feature_size = feature_size
        
        # 使用自适应池化
        self.ch_AvgPool = nn.AdaptiveAvgPool3d(1)
        self.ch_MaxPool = nn.AdaptiveMaxPool3d(1)
        
        # 通道注意力部分
        self.ch_Linear1 = nn.Linear(in_planes, in_planes // 4, bias=False)
        self.ch_Linear2 = nn.Linear(in_planes // 4, in_planes, bias=False)
        self.ch_Sigmoid = nn.Sigmoid()

        # 空间注意力部分
        self.sp_Conv = nn.Conv3d(2, 1, kernel_size=5, stride=1, padding=2, bias=False)  # 使用更大的卷积核
        self.sp_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力部分
        x_ch_avg_pool = self.ch_AvgPool(x).view(x.size(0), -1)
        x_ch_max_pool = self.ch_MaxPool(x).view(x.size(0), -1)

        x_ch_avg_linear = self.ch_Linear2(self.ch_Linear1(x_ch_avg_pool))
        x_ch_max_linear = self.ch_Linear2(self.ch_Linear1(x_ch_max_pool))

        ch_out = (self.ch_Sigmoid(x_ch_avg_linear + x_ch_max_linear).view(x.size(0), self.in_planes, 1, 1, 1)) * x

        # 空间注意力部分
        x_sp_max_pool = torch.max(ch_out, 1, keepdim=True)[0]
        x_sp_avg_pool = torch.mean(ch_out, 1, keepdim=True)

        sp_conv1 = torch.cat([x_sp_max_pool, x_sp_avg_pool], dim=1)
        sp_out = self.sp_sigmoid(self.sp_Conv(sp_conv1))

        # 融合注意力
        out = sp_out * x + x  # 残差连接
        return out




def make_conv3d(in_channels: int, out_channels: int, kernel_size: typing.Union[int, tuple], stride: int,
                padding: int, dilation=1, groups=1,
                bias=True) -> nn.Module:
    """
    produce a Conv3D with Batch Normalization and ReLU

    :param in_channels: num of in in
    :param out_channels: num of out channels
    :param kernel_size: size of kernel int or tuple
    :param stride: num of stride
    :param padding: num of padding
    :param bias: bias
    :param groups: groups
    :param dilation: dilation
    :return: my conv3d module
    """
    module = nn.Sequential(

        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  groups=groups,
                  bias=bias),
        nn.BatchNorm3d(out_channels),
        nn.ReLU())
    return module


def conv3d_same_size(in_channels, out_channels, kernel_size, stride=1,
                     dilation=1, groups=1,
                     bias=True):
    padding = kernel_size // 2
    return make_conv3d(in_channels, out_channels, kernel_size, stride,
                       padding, dilation, groups,
                       bias)


def conv3d_pooling(in_channels, kernel_size, stride=1,
                   dilation=1, groups=1,
                   bias=False):
    padding = kernel_size // 2
    return make_conv3d(in_channels, in_channels, kernel_size, stride,
                       padding, dilation, groups,
                       bias)


class ResidualBlock(nn.Module):
    """
    a simple residual block
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.my_conv1 = make_conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.my_conv2 = make_conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = make_conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        out1 = self.conv3(inputs)
        out = self.my_conv1(inputs)
        out = self.my_conv2(out)
        out = out + out1
        return out

class SegmentationNet(nn.Module):
    def __init__(self):
        super(SegmentationNet, self).__init__()
        self.conv1 = conv3d_same_size(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = conv3d_same_size(in_channels=8, out_channels=16, kernel_size=3)
        self.conv3 = conv3d_same_size(in_channels=16, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool3d(kernel_size=2)
        
        # 增加到输出形状为(512, 2, 8, 8)
        self.final_conv = nn.Conv3d(in_channels=32, out_channels=512, kernel_size=1)  # 1x1x1卷积

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.final_conv(x)  # 输出为 (batch_size, 512, d, h, w)
        return x

class ConvRes_table(nn.Module):
    def __init__(self, config, categories, num_special_tokens=2):
        super(ConvRes_table, self).__init__()
        self.conv1 = conv3d_same_size(in_channels=1, out_channels=4, kernel_size=3)
        self.conv2 = conv3d_same_size(in_channels=4, out_channels=4, kernel_size=3)
        
        self.config = config
        self.last_channel = 4
        self.first_cbam = ResCBAMLayer(4, (16,64,64))
        layers = []
        i = 0
        for stage in config:
            i = i+1
            layers.append(conv3d_pooling(self.last_channel, kernel_size=3, stride=2))
            for channel in stage:
                layers.append(ResidualBlock(self.last_channel, channel))
                self.last_channel = channel
                #layers.append(CoordAttention(feature_size=self.last_channel, coord_size=3))
            layers.append(ResCBAMLayer(self.last_channel, (16//(2**i),64//(2**i),64//(2**i))))
        self.layers = nn.Sequential(*layers)
        self.avg_pooling = nn.AvgPool3d(kernel_size=(2,8,8), stride=4)
        self.linear_proj = nn.Linear(512, 128)  # d_embed 为原始嵌入维度
        transformer_heads = 16
        self.cls_token = nn.Parameter(torch.randn(1, 1, config[-1][-1]*2))
        self.tep = Transformer(dim=config[-1][-1]*2, heads=transformer_heads, depth=2, 
                               attn_dropout=0.1, ff_dropout=0.1, dim_head=config[-1][-1]*2 // transformer_heads)
        self.tepfushon = TemporalFusion(feature_size=512)
        self.fc = nn.Linear(config[-1][-1]*2, out_features=2)

        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, config[-1][-1])

        self.cross_atten = CrossAttention(n_heads=8, d_embed=config[-1][-1]*2, d_cross=config[-1][-1]) # all hard code to avoid complex parameter settings
        self.cross_feed = FeedForward(config[-1][-1]*2, mult=2, dropout=0.1) # all hard code to avoid complex parameter settings
        #self.cross_fusion = MultimodalModel(img_feat_dim=128, text_feat_dim=128, fusion_dim=128)
        self.fusion = CMFA(img_dim=128,tab_dim=128,hid_dim=128)
        self.fc2 = nn.Linear(512,256)
        self.ai1 = nn.ReLU()
        self.fc3 = nn.Linear(256,128) 
        self.c1 = nn.Linear(256,128)
        self.a1 = nn.ReLU()
        self.c2 = nn.Linear(128,2)


    #def forward(self, input1, input2, coords1, coords2, seg1, seg2):
    def forward(self, input1, input2, text_data):    
        out1 = self.conv1(input1)
        out1 = self.conv2(out1)
        out1 = self.first_cbam(out1)
        out1 = self.layers(out1)
        
        # Process the second image
        out2 = self.conv1(input2)
        out2 = self.conv2(out2)
        out2 = self.first_cbam(out2)
        out2 = self.layers(out2)

        # deal with table data
        assert text_data.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        x_categ = text_data + self.categories_offset
        #print(x_categ.shape)
        x_categ = self.categorical_embeds(x_categ)
        x_categ = x_categ.squeeze(1)  # 去掉多余的维度 (B, num_categories, d_embed)
        #print(x_categ.shape)
        x_categ = self.linear_proj(x_categ)  # (B, num_categories, 128)
        #print(x_categ.shape)

        x_categ = x_categ.mean(dim=1)  # 或者 x_categ.sum(dim=1) 根据需求
        #print(x_categ.shape)

        #out = self.tepfushon(out1, out2)
        out = self.tepfushon(out1, out2)
        out = self.avg_pooling(out)
        out = out.view(out.size(0), -1)
        #out = self.tep(out)
        out = self.fc2(out)
        out = self.ai1(out)
        out = self.fc3(out)
        
        #out = self.cross_fusion(out, x_categ)
        #out = self.fusion(out, x_categ)
        #print(out.shape)
        fusion = self.fusion(out, x_categ)
        out = self.c1(fusion)
        out = self.a1(out)
        out = self.c2(out)        
        
        return out


class TemporalFusion(nn.Module):
    def __init__(self, feature_size):
        super(TemporalFusion, self).__init__()
        
        # t1 特征处理: 上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        # 自适应平均池化
        self.avgpool_t1 = nn.AdaptiveAvgPool3d((2, 8, 8))  # t1 的平均池化
        self.avgpool_t0 = nn.AdaptiveAvgPool3d((2, 8, 8))  # t0 的平均池化
        
        # 3D卷积
        self.conv3d = nn.Conv3d(in_channels=1024, out_channels=feature_size, kernel_size=3, padding=1)
        
        # 残差层：继续使用卷积代替Linear
        self.res_conv1 = nn.Conv3d(in_channels=feature_size, out_channels=feature_size, kernel_size=3, padding=1)
        self.res_conv2 = nn.Conv3d(in_channels=feature_size, out_channels=feature_size, kernel_size=3, padding=1)
        
        # 可学习权重
        self.weight_t0 = nn.Parameter(torch.tensor(0.3))  # t0 的权重
        self.weight_t1 = nn.Parameter(torch.tensor(0.7))  # t1 的权重
        
        self.relu = nn.ReLU()

    def forward(self, t0, t1):
        # t1 特征的处理: 上采样 -> 平均池化 -> 3D卷积
        fush = torch.cat([t0, t1], dim=1)
        fush = self.upsample(fush)  # t1 上采样
        fush = self.avgpool_t1(fush)  # t1 平均池化
        fush = self.conv3d(fush)  # t1 3D卷积

        # 第一层残差连接（保留3D特征形状，使用卷积）
        #t1_fused = self.relu(t1_conv + self.res_conv1(t1_conv))  # 残差连接

        # t0 进行平均池化
        t0_pooled = self.avgpool_t0(t0)  # t0 平均池化

        # 可学习加权融合（保持3D形状进行加权融合）
        weight_t0 = torch.sigmoid(self.weight_t0)
        weight_t1 = torch.sigmoid(self.weight_t1)
        fused_feature = weight_t0 * fush + weight_t1 * t1  # 加权融合

        return fused_feature




def test():
    global debug
    debug = True
    net = ConvResRFCBAM([[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]])
    inputs = torch.randn((1, 1, 16, 64, 64))
    output = net(inputs)
    print(net)
    print(output.shape)
    
if __name__ == '__main__':
    test()
