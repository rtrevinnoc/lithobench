import torch
import torch.nn as nn


class MiniUNet(nn.Module):
    """
    Scaled-down UNet for FPGA deployment via hls4ml.

    Architecture mirrors the original NeuralILT UNet (lithobench/ilt/neuralilt.py)
    but with reduced channel widths for on-chip FPGA resource constraints.

    Original: 1 -> 64 -> 128 -> 256 -> 512  (~7.8M params)
    This:     1 ->  8 ->  16 ->  32 ->  64  (~122K params)

    Changes from original:
    - nn.Upsample(scale_factor=2, mode='nearest') for upsampling (parameter-free,
      exports to ONNX Resize op which hls4ml's ONNX frontend supports)
    - Explicit nn.Sequential blocks (no helper functions) for torch.fx tracing
    - 1x1 final conv instead of 3x3

    Input:  (B, 1, 64, 64)  — single-channel tile
    Output: (B, 1, 64, 64)  — predicted mask tile
    """

    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Decoder — nearest-neighbour upsample (exports as ONNX Resize, supported by hls4ml)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv3 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv2 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv1 = nn.Sequential(
            nn.Conv2d(16 + 8, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.final_conv = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)         # (B, 8, 64, 64)
        x = self.pool1(c1)         # (B, 8, 32, 32)

        c2 = self.conv2(x)         # (B, 16, 32, 32)
        x = self.pool2(c2)         # (B, 16, 16, 16)

        c3 = self.conv3(x)         # (B, 32, 16, 16)
        x = self.pool3(c3)         # (B, 32, 8, 8)

        # Bottleneck
        x = self.conv4(x)          # (B, 64, 8, 8)

        # Decoder with skip connections
        x = self.up3(x)            # (B, 64, 16, 16)
        x = torch.cat([x, c3], dim=1)  # (B, 96, 16, 16)
        x = self.deconv3(x)        # (B, 32, 16, 16)

        x = self.up2(x)            # (B, 32, 32, 32)
        x = torch.cat([x, c2], dim=1)  # (B, 48, 32, 32)
        x = self.deconv2(x)        # (B, 16, 32, 32)

        x = self.up1(x)            # (B, 16, 64, 64)
        x = torch.cat([x, c1], dim=1)  # (B, 24, 64, 64)
        x = self.deconv1(x)        # (B, 8, 64, 64)

        x = self.final_conv(x)     # (B, 1, 64, 64)
        x = self.sigmoid(x)

        return x


def fuse_batchnorm(model):
    """Fuse BatchNorm layers into preceding Conv2d for inference / hls4ml export.

    Returns a new model with BN folded into conv weights and biases.
    The original model is not modified.
    """
    import copy
    fused = copy.deepcopy(model)
    fused.eval()

    # Walk named children and replace Sequential blocks with fused versions
    for attr_name, module in list(fused.named_children()):
        if not isinstance(module, nn.Sequential):
            continue

        layers = list(module.children())
        new_layers = []
        i = 0
        while i < len(layers):
            if (
                i + 1 < len(layers)
                and isinstance(layers[i], nn.Conv2d)
                and isinstance(layers[i + 1], nn.BatchNorm2d)
            ):
                fused_conv = _fuse_conv_bn(layers[i], layers[i + 1])
                new_layers.append(fused_conv)
                i += 2
            else:
                new_layers.append(layers[i])
                i += 1

        # Replace with a rebuilt Sequential
        setattr(fused, attr_name, nn.Sequential(*new_layers))

    return fused


def _fuse_conv_bn(conv, bn):
    """Fuse a Conv2d and BatchNorm2d into a single Conv2d."""
    with torch.no_grad():
        # BN parameters
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        std = torch.sqrt(var + eps)
        scale = gamma / std

        # Fused weight = conv.weight * scale (broadcast over output channels)
        fused_weight = conv.weight * scale.reshape(-1, 1, 1, 1)

        # Fused bias
        if conv.bias is not None:
            fused_bias = (conv.bias - mean) * scale + beta
        else:
            fused_bias = -mean * scale + beta

        fused_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        fused_conv.weight.copy_(fused_weight)
        fused_conv.bias.copy_(fused_bias)

    return fused_conv
