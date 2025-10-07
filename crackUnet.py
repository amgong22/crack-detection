import torch
import torch.nn as nn


# --- Helper Block: Two Conv Layers + BN + LeakyReLU ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# --- Main U-Net Architecture ---
class CrackUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder (Downsampling)
        self.enc1 = DoubleConv(in_channels, 64, dropout=0.1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128, dropout=0.1)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256, dropout=0.2)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512, dropout=0.3)

        # Decoder (Upsampling)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256, dropout=0.2)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128, dropout=0.1)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64, dropout=0.1)

        # Output Layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # Initialize weights
        self._init_weights()

    # --- Forward Pass ---
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Output probability map
        return torch.sigmoid(self.out_conv(d1))

    # --- Weight Initialization ---
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# --- Quick Test (optional) ---
if __name__ == "__main__":
    model = CrackUNet(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 256, 256)  # test input
    y = model(x)
    print("Output shape:", y.shape)
