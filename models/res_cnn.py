import typing

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

# SE (Squeeze-and-Excitation) module with 3D inputs
class SE3D(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(SE3D, self).__init__()
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))  # 3D pooling
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // ratio, bias=False),  # From c -> c/r
            nn.ReLU(),
            nn.Linear(in_channel // ratio, in_channel, bias=False),  # From c/r -> c
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c = x.shape[0:2]
        y = self.gap(x).view(b, c)  # Global average pooling
        y = self.fc(y).view(b, c, 1, 1, 1)  # Re-shaping back to (b, c, 1, 1, 1)
        return y


class ResRFCBAMLayer(nn.Module):
    def __init__(self, in_planes, feature_size, kernel_size=3):
        super(ResRFCBAMLayer, self).__init__()
        self.in_planes = in_planes
        self.feature_size = feature_size
        self.kernel_size = kernel_size

        # 使用 SE 模块来替代 CBAM 中的通道注意力机制
        self.se = SE3D(in_planes)
        
        # 使用自适应感受野注意力代替 CBAM 的空间注意力机制
        self.generate = nn.Sequential(
            nn.Conv3d(in_planes, in_planes * (kernel_size**3), kernel_size, padding=kernel_size//2,
                      stride=1, dilation=1, groups=in_planes, bias=False),
            nn.BatchNorm3d(in_planes * (kernel_size**3)),
            nn.ReLU()
        )
        
        self.get_weight = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False), 
            nn.Sigmoid()
        )
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_planes, in_planes, kernel_size, stride=kernel_size)
        )

    def forward(self, x):
        b, c = x.shape[0:2]

        # 1. 通道注意力部分：使用 SE (Squeeze-and-Excitation)
        channel_attention = self.se(x)

        # 2. 生成特征：使用深度可分离卷积生成自适应感受野特征
        generate_feature = self.generate(x)

        d, h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b, c, self.kernel_size**3, d, h, w)

        # 3. 展开特征：rearrange 到对应维度
        generate_feature = rearrange(generate_feature, 'b c (n1 n2 n3) d h w -> b c (d n1) (h n2) (w n3)', 
                                     n1=self.kernel_size, n2=self.kernel_size, n3=self.kernel_size)
        generate_feature = generate_feature * channel_attention.view(b, c, 1, 1, 1)
        # 4. 感受野注意力
        max_feature, _ = torch.max(generate_feature, dim=1, keepdim=True)
        mean_feature = torch.mean(generate_feature, dim=1, keepdim=True)
        
        receptive_field_attention = self.get_weight(torch.cat((max_feature, mean_feature), dim=1))

        # 5. 融合特征
        conv_data = generate_feature * receptive_field_attention
        conv_out = self.conv(conv_data)

        # 6. 残差连接：将输出和原始输入相加
        out = conv_out + x

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


class ConvRes(nn.Module):
    def __init__(self, config):
        super(ConvRes, self).__init__()
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
        self.segmentation_net = SegmentationNet()
        #self.coord_attention = CoordAttention(feature_size=4, coord_size=3)
        self.avg_pooling = nn.AvgPool3d(kernel_size=(2,8,8), stride=4)
        #self.tep = TemporalFusionModel(feature_dim=512, num_heads=8, num_layers=6)
        #self.tep = TemporalTransformerModel(feature_dim=128, num_heads=8, num_layers=2, output_dim=2)
        self.tep = TemporalFusion(feature_size=512)
        self.fc = AngleLinear(in_features=512, out_features=2)
        self.fusion = CMFA(img_dim=128,tab_dim=128,hid_dim=128)
        self.fc2 = nn.Linear(512,256)
        self.ai1 = nn.ReLU()
        self.fc3 = nn.Linear(256,128) 
        self.c1 = nn.Linear(256,128)
        self.a1 = nn.ReLU()
        self.c2 = nn.Linear(128,2)      
        # 高斯权重图层
        self.gaussian_layer = GaussianHeatmapLayer(height=64, width=64, depth=16)
        # 初始化空间注意力机制
        self.spatial_attention = SpatialAttention3D(in_channels=4)  # 注意这里的输入通道数  
    #def forward(self, input1, input2, coords1, coords2, seg1, seg2):
    def forward(self, input1, input2):    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #if input1.size(0) == 4:
        #    coords = coords1
        #elif input1.size(0) == 2:
        #    coords = coords2
       # elif input1.size(0) == 1:
      #      coords = torch.tensor([[8.0, 32.0, 32.0]])
       # else:
       #     coords = torch.tensor([[8.0, 32.0, 32.0],
       #                            [8.0, 32.0, 32.0],
       #                            [8.0, 32.0, 32.0]])
        coords = torch.tensor([8.0, 32.0, 32.0]).to(device)
        coords = coords.repeat(input1.size(0), 1)  # 重复batch_size次
        # 生成与特征图同尺寸的高斯权重图
        #gaussian_map = self.gaussian_layer(coords).to(device)  # (batch_size, 1, depth, height, width)
        #gaussian_map = gaussian_map.to(device)
        #coords = coords.to(device)
        #out = self.conv2(out)
        # out = self.conv1(inputs) #stage1
        #out = self.first_cbam(out) #stage2
        
        # Process the first image
        
        out1 = self.conv1(input1)
        out1 = self.conv2(out1)
        out1 = self.first_cbam(out1)
        #out1 = out1.to(device)
        #out1 = out1 * gaussian_map
        #out1 = self.spatial_attention(out1, coords)  # 传入三维坐标
        #out1 = self.coord_attention(out1, coords)
        #print(out1.shape)
        out1 = self.layers(out1)
        #out1 = self.layers[0](out1)  # 第一层的特征
        #for i, layer in enumerate(self.layers[1:], 1):
            # 对每个ResidualBlock之后的CoordAttention层传入coords
            #if isinstance(layer, CoordAttention):
                #out1 = layer(out1, coords)
           # else:
               # out1 = layer(out1)

        
        
        # Process the second image
        out2 = self.conv1(input2)
        out2 = self.conv2(out2)
        out2 = self.first_cbam(out2)
        #out2 = out2.to(device)
        #out2 = out2 * gaussian_map
        #out2 = self.spatial_attention(out2, coords)  # 传入三维坐标
        out2 = self.layers(out2)
        
        #out2 = self.layers[0](out2)  # 第一层的特征
        #for i, layer in enumerate(self.layers[1:], 1):
            # 对每个ResidualBlock之后的CoordAttention层传入coords
            #if isinstance(layer, CoordAttention):
               # out2 = layer(out2, coords)
            #else:
               # out2 = layer(out2)
        # 处理分割文件
        #seg_out1 = self.segmentation_net(seg1)  # 输出形状为 (4, 512, 2, 8, 8)
        #seg_out2 = self.segmentation_net(seg2)  # 输出形状为 (4, 512, 2, 8, 8)
        
        #融合分割与图像特征
        #fused1 = torch.cat([out1, seg_out1], dim=1)  # 融合第一个图像和分割
        #fused2 = torch.cat([out2, seg_out2], dim=1)  # 融合第二个图像和分割

        
        # 将高斯权重图与特征图逐点相乘
        
        
        #out1 = self.avg_pooling(out1)
        #out2 = self.avg_pooling(out2)
        #out1 = out1.view(out1.size(0), -1)
        #out2 = out2.view(out2.size(0), -1)
        #out1 = self.fc2(out1)
        #out2 = self.fc2(out2)
        #out1 = self.ai1(out1)
        #out2 = self.ai1(out2)
        #out1 = self.fc3(out1)
        #out2 = self.fc3(out2)
        #fusion = self.fusion(out1, out2)
        #out = self.c1(fusion)
        #out = self.a1(out)
        #out = self.c2(out)
        # 进一步融合两个结果
        out = self.tep(out1, out2)
        
        out = self.avg_pooling(out)
        
        out = out.view(out.size(0), -1)
        #out = self.tep(out1, out2)
        
        out = self.fc(out)
        #print(out.shape)
        return out


class GaussianHeatmapLayer(nn.Module):
    def __init__(self, height, width, depth):
        super(GaussianHeatmapLayer, self).__init__()
        self.height = height
        self.width = width
        self.depth = depth
        self.coord_offset = nn.Parameter(torch.tensor([4.0, 20.0, 20.0]))  # z, y, x 的初始偏移量

    def forward(self, coords):
        batch_size = coords.size(0)
        device = coords.device
        coords = coords + self.coord_offset.to(device)

        z_coords, y_coords, x_coords = coords[:, 0], coords[:, 1], coords[:, 2]
        gaussian_maps = torch.zeros((batch_size, 1, self.depth, self.height, self.width), device=device)

        for i in range(batch_size):
            z_center, y_center, x_center = z_coords[i].item(), y_coords[i].item(), x_coords[i].item()

            z_grid, y_grid, x_grid = torch.meshgrid(
                torch.arange(self.depth, device=device),
                torch.arange(self.height, device=device),
                torch.arange(self.width, device=device),
                indexing='ij'
            )

            # 设置 sigma 为合适的值，控制影响范围
            sigma_z = 3.0  # z轴的标准差
            sigma_xy = 12.0  # x和y轴的标准差

            # 计算高斯热图，考虑到偏移范围
            gaussian_map = torch.exp(
                -((z_grid - z_center)**2 / (2 * sigma_z**2) +
                  (y_grid - y_center)**2 / (2 * sigma_xy**2) +
                  (x_grid - x_center)**2 / (2 * sigma_xy**2))
            )

            # 将计算得到的热图赋值给 gaussian_maps
            gaussian_maps[i, 0] = gaussian_map

        return gaussian_maps
    
# 定义三维空间注意力机制
class SpatialAttention3D(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1)
        self.conv3 = nn.Conv3d(in_channels // 2, 1, kernel_size=1)
        self.coords_adjust = nn.Conv3d(3, in_channels // 2, kernel_size=1)
    def forward(self, x, coords):
        # x: 输入特征图，形状为 (batch_size, channels, depth, height, width)
        # coords: 三维坐标信息，形状为 (batch_size, 3)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        print(x.shape)
        batch_size, channels, depth, height, width = x.size()
        coords = coords.to(device)
        #print(batch_size)
        # 将坐标归一化到特征图的空间尺寸
        z = torch.tensor([depth, height, width], dtype=torch.float32).to(device)
        coords_norm = (coords / z)
        
        # 计算空间注意力权重
        attention_map = self.conv1(x) + self.conv2(x)
        #print(attention_map.shape)
        # 使用坐标信息调整注意力图
        coords_att = coords_norm.view(batch_size, 3, 1, 1, 1)  # 将坐标转换为 (batch_size, 3, 1, 1, 1)
        coords_att = F.interpolate(coords_att, size=(depth, height, width), mode='trilinear', align_corners=False)
        coords_att = self.coords_adjust(coords_att)  # 变换为 (batch_size, channels // 2, depth, height, width)
        #print(coords_att.shape)
        # 将坐标信息与注意力图结合
        attention_map = attention_map + coords_att

        # 通过sigmoid激活函数生成注意力权重
        attention_weights = torch.sigmoid(self.conv3(attention_map))

        # 将注意力权重应用于输入特征图
        out = x * attention_weights

        return out

class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model, height, width, depth):
        super(PositionalEncoding3D, self).__init__()
        self.d_model = d_model

        # 创建位置编码
        pe = torch.zeros(d_model, depth, height, width)
        position_h = torch.arange(0, height, dtype=torch.float).unsqueeze(1)
        position_w = torch.arange(0, width, dtype=torch.float).unsqueeze(1)
        position_d = torch.arange(0, depth, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))

        pe[0::2, :, :, :] = torch.sin(position_d * div_term).unsqueeze(1).unsqueeze(1)  # z轴
        pe[1::2, :, :, :] = torch.cos(position_d * div_term).unsqueeze(1).unsqueeze(1)  # z轴

        pe[0::2, :, :, :] += torch.sin(position_h * div_term).unsqueeze(1)  # y轴
        pe[1::2, :, :, :] += torch.cos(position_h * div_term).unsqueeze(1)  # y轴

        pe[0::2, :, :, :] += torch.sin(position_w * div_term)  # x轴
        pe[1::2, :, :, :] += torch.cos(position_w * div_term)  # x轴

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # 将位置编码添加到输入特征
        x = x + self.pe[:, :x.size(1), :x.size(2), :x.size(3), :x.size(4)]
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

class ConvResRFCBAM_table(nn.Module):
    def __init__(self, config, categories, num_special_tokens=2):
        super(ConvResRFCBAM_table, self).__init__()
        self.conv1 = conv3d_same_size(in_channels=1, out_channels=4, kernel_size=3)
        self.conv2 = conv3d_same_size(in_channels=4, out_channels=4, kernel_size=3)
        
        self.config = config
        self.last_channel = 4
        self.first_cbam = ResRFCBAMLayer(4, (16,64,64))
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
        self.temporalfusion = TemporalFusion(feature_size=512)
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
        out = self.temporalfusion(out1, out2)
        
        out = self.avg_pooling(out)
        out = out.view(out.size(0), -1)
        out = self.fc2(out)
        out = self.ai1(out)
        out = self.fc3(out)
        #print(out.shape)
        fusion = self.fusion(out, x_categ)
        out = self.c1(fusion)
        out = self.a1(out)
        out = self.c2(out)        
        
        return out


class ConvResRFCBAM(nn.Module):
    def __init__(self, config):
        super(ConvResRFCBAM, self).__init__()
        self.conv1 = conv3d_same_size(in_channels=1, out_channels=4, kernel_size=3)
        self.conv2 = conv3d_same_size(in_channels=4, out_channels=4, kernel_size=3)
        self.config = config
        self.last_channel = 4
        self.first_cbam = ResRFCBAMLayer(4, (16,64,64))
        self.coord_attention = CoordAttention(feature_size=self.last_channel, coord_size=3)
        layers = []
        i = 0
        for stage in config:
            i = i+1
            layers.append(conv3d_pooling(self.last_channel, kernel_size=3, stride=2))
            for channel in stage:
                layers.append(ResidualBlock(self.last_channel, channel))
                self.last_channel = channel
            layers.append(ResCBAMLayer(self.last_channel, (16//(2**i),64//(2**i),64//(2**i))))
        self.layers = nn.Sequential(*layers)
        self.segmentation_net = SegmentationNet()
        self.avg_pooling = nn.AvgPool3d(kernel_size=(2,8,8), stride=4)
        self.fc = AngleLinear(in_features=512, out_features=2)

    def forward(self, inputs):
        if debug:
            print(inputs.size())
        out = self.conv1(inputs)
        if debug:
            print(out.size())
        out = self.conv2(out)
        if debug:
            print(out.size())
        out = self.first_cbam(out)
        out = self.layers(out)
        print('outz:',out.shape)

        
        if debug:
            print(out.size())
        out = self.avg_pooling(out)
        out = out.view(out.size(0), -1)
        if debug:
            print(out.size())
        #out = self.att(out,coords)
        out = self.fc(out)
        #print(out.shape)
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

class ConvRes_table_onlyt0(nn.Module):
    def __init__(self, config, categories, num_special_tokens=2):
        super(ConvRes_table_onlyt0, self).__init__()
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
        self.cross_fusion = MultimodalModel(img_feat_dim=128, text_feat_dim=128, fusion_dim=128)
        self.fusion = CMFA(img_dim=128,tab_dim=128,hid_dim=128)
        self.fc2 = nn.Linear(512,256)
        self.ai1 = nn.ReLU()
        self.fc3 = nn.Linear(256,128) 
        self.c1 = nn.Linear(256,128)
        self.a1 = nn.ReLU()
        self.c2 = nn.Linear(128,2)


    #def forward(self, input1, input2, coords1, coords2, seg1, seg2):
    def forward(self, input1, text_data):    
        out = self.conv1(input1)
        out = self.conv2(out)
        out = self.first_cbam(out)
        out = self.layers(out)
        
        # Process the second image
        #out2 = self.conv1(input2)
        #out2 = self.conv2(out2)
        #out2 = self.first_cbam(out2)
        #out2 = self.layers(out2)

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
        #out = self.tepfushon(out1, out2)
        out = self.avg_pooling(out)
        out = out.view(out.size(0), -1)
        #out = self.tep(out)
        out = self.fc2(out)
        out = self.ai1(out)
        out = self.fc3(out)
        
        out = self.cross_fusion(out, x_categ)
        
        #print(out.shape)
        #fusion = self.fusion(out, x_categ)
        #out = self.c1(fusion)
        #out = self.a1(out)
        #out = self.c2(out)        
        
        return out

class ConvRes_onlyt1(nn.Module):
    def __init__(self, config, categories, num_special_tokens=2):
        super(ConvRes_onlyt1, self).__init__()
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
        self.fc = nn.Linear(config[-1][-1], out_features=2)

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
        self.cross_fusion = MultimodalModel(img_feat_dim=128, text_feat_dim=128, fusion_dim=128)
        self.fusion = CMFA(img_dim=128,tab_dim=128,hid_dim=128)
        self.fc2 = nn.Linear(512,256)
        self.ai1 = nn.ReLU()
        self.fc3 = nn.Linear(256,128) 
        self.c1 = nn.Linear(256,128)
        self.a1 = nn.ReLU()
        self.c2 = nn.Linear(128,2)


    #def forward(self, input1, input2, coords1, coords2, seg1, seg2):
    def forward(self, input1):    
        out = self.conv1(input1)
        out = self.conv2(out)
        out = self.first_cbam(out)
        out = self.layers(out)
        
        # Process the second image
        #out2 = self.conv1(input2)
        #out2 = self.conv2(out2)
        #out2 = self.first_cbam(out2)
        #out2 = self.layers(out2)

        # deal with table data
        #assert text_data.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        #x_categ = text_data + self.categories_offset
        #print(x_categ.shape)
        #x_categ = self.categorical_embeds(x_categ)
        #x_categ = x_categ.squeeze(1)  # 去掉多余的维度 (B, num_categories, d_embed)
        #print(x_categ.shape)
        #x_categ = self.linear_proj(x_categ)  # (B, num_categories, 128)
        #print(x_categ.shape)

        #x_categ = x_categ.mean(dim=1)  # 或者 x_categ.sum(dim=1) 根据需求
        #print(x_categ.shape)

        #out = self.tepfushon(out1, out2)
        #out = self.tepfushon(out1, out2)
        out = self.avg_pooling(out)
        out = out.view(out.size(0), -1)
        #out = self.tep(out)
        #out = self.fc2(out)
        #out = self.ai1(out)
       #out = self.fc3(out)
        
        #out = self.cross_fusion(out, x_categ)
        
        #print(out.shape)
        #fusion = self.fusion(out, x_categ)
        #out = self.c1(fusion)
        #out = self.a1(out)
        #out = self.c2(out)        
        out = self.fc(out)
        return out

class ConvRes_table_onlyt1(nn.Module):
    def __init__(self, config, categories, num_special_tokens=2):
        super(ConvRes_table_onlyt1, self).__init__()
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
        self.fc = nn.Linear(config[-1][-1], out_features=2)

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
        self.cross_fusion = MultimodalModel(img_feat_dim=128, text_feat_dim=128, fusion_dim=128)
        self.fusion = CMFA(img_dim=128,tab_dim=128,hid_dim=128)
        self.fc2 = nn.Linear(512,256)
        self.ai1 = nn.ReLU()
        self.fc3 = nn.Linear(256,128) 
        self.c1 = nn.Linear(256,128)
        self.a1 = nn.ReLU()
        self.c2 = nn.Linear(128,2)


    #def forward(self, input1, input2, coords1, coords2, seg1, seg2):
    def forward(self, input1, text_data):    
        out = self.conv1(input1)
        out = self.conv2(out)
        out = self.first_cbam(out)
        out = self.layers(out)
        
        # Process the second image
        #out2 = self.conv1(input2)
        #out2 = self.conv2(out2)
        #out2 = self.first_cbam(out2)
        #out2 = self.layers(out2)

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
        #out = self.tepfushon(out1, out2)
        out = self.avg_pooling(out)
        out = out.view(out.size(0), -1)
        #out = self.tep(out)
        out = self.fc2(out)
        out = self.ai1(out)
        out = self.fc3(out)
        #print(out.shape)
        #out = self.cross_fusion(out, x_categ)
        
        #print(out.shape)
        fusion = self.fusion(out, x_categ)
        out = self.c1(fusion)
        out = self.a1(out)
        out = self.c2(out)        
        #out = self.fc(out)
        return out

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