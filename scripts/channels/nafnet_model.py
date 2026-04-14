# 尝试导入 PyTorch
torch_available = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch_available = True
except ImportError:
    pass

if torch_available:
    class LayerNorm2d(nn.Module):
        """LayerNorm for 2D images"""
        def __init__(self, num_features, eps=1e-6):
            super(LayerNorm2d, self).__init__()
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
            self.eps = eps
        
        def forward(self, x):
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    
    class SimplifiedCA(nn.Module):
        """Simplified Channel Attention"""
        def __init__(self, channels):
            super(SimplifiedCA, self).__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        def forward(self, x):
            y = self.pool(x)
            y = self.conv(y)
            y = torch.sigmoid(y)
            return x * y
    
    class NAFBlock(nn.Module):
        """NAFNet Block"""
        def __init__(self, c, expansion=2):
            super(NAFBlock, self).__init__()
            self.dwconv = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=1)  # 使用 groups=1 避免通道数不匹配
            self.norm = LayerNorm2d(c)
            self.pwconv1 = nn.Conv2d(c, expansion * c, kernel_size=1)
            self.pwconv2 = nn.Conv2d(expansion * c, c, kernel_size=1)
            self.sca = SimplifiedCA(expansion * c)
        
        def forward(self, x):
            residual = x
            x = self.dwconv(x)
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.sca(x)
            x = self.pwconv2(x)
            return x + residual
    
    class NAFNet(nn.Module):
        """NAFNet model"""
        def __init__(self, in_channels=3, out_channels=3, width=64, num_blocks=[2, 2, 2, 2], num_channels=None):
            super(NAFNet, self).__init__()
            
            # 如果没有提供 num_channels，根据 width 生成
            if num_channels is None:
                num_channels = [width, width*2, width*4, width*8]
            
            # 下采样
            self.downsample = nn.ModuleList()
            self.encoders = nn.ModuleList()
            
            # 输入卷积
            self.conv_in = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
            
            # 编码器
            for i in range(len(num_blocks)):
                # 块
                blocks = nn.ModuleList()
                for j in range(num_blocks[i]):
                    blocks.append(NAFBlock(num_channels[i]))
                self.encoders.append(blocks)
                
                # 下采样（除了最后一层）
                if i < len(num_blocks) - 1:
                    self.downsample.append(nn.Conv2d(num_channels[i], num_channels[i+1], kernel_size=2, stride=2))
            
            # 解码器
            self.upsample = nn.ModuleList()
            self.decoders = nn.ModuleList()
            
            for i in range(len(num_blocks)-1, -1, -1):
                # 块
                blocks = nn.ModuleList()
                for j in range(num_blocks[i]):
                    blocks.append(NAFBlock(num_channels[i]))
                self.decoders.append(blocks)
                
                # 上采样（除了最后一层）
                if i > 0:
                    self.upsample.append(nn.ConvTranspose2d(num_channels[i], num_channels[i-1], kernel_size=2, stride=2))
            
            # 输出卷积
            self.conv_out = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)
        
        def forward(self, x):
            # 输入
            x = self.conv_in(x)
            
            # 编码
            encoder_features = []
            for i, blocks in enumerate(self.encoders):
                # 应用块
                for block in blocks:
                    x = block(x)
                encoder_features.append(x)
                
                # 下采样（除了最后一层）
                if i < len(self.encoders) - 1:
                    x = self.downsample[i](x)
            
            # 解码
            for i in range(len(self.decoders)):
                # 应用块
                for block in self.decoders[i]:
                    x = block(x)
                
                # 上采样（除了最后一层）
                if i < len(self.upsample):
                    x = self.upsample[i](x)
                    # 残差连接，确保尺寸匹配
                    encoder_feature = encoder_features[len(encoder_features)-2-i]
                    if x.shape != encoder_feature.shape:
                        # 使用插值调整尺寸
                        encoder_feature = F.interpolate(
                            encoder_feature, 
                            size=(x.shape[2], x.shape[3]), 
                            mode='bilinear', 
                            align_corners=False
                        )
                    x = x + encoder_feature
            
            # 输出
            x = self.conv_out(x)
            return x

def load_nafnet_model(model_path=None):
    """加载NAFNet模型
    
    Args:
        model_path: 模型权重路径
        
    Returns:
        NAFNet: 加载权重的模型
    """
    if not torch_available:
        print("PyTorch 未安装，无法加载 NAFNet 模型")
        return None
    
    # 创建模型
    model = NAFNet()
    
    # 加载权重
    if model_path and torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    elif model_path:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # 设置为评估模式
    model.eval()
    
    return model
