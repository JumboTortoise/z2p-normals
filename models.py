from networks import *
import torch.nn.functional as F
import torch


class PosADANet(nn.Module):
    def encode(self, shp):
        device = self.omega.device
        B, _, H, W = shp
        row = torch.arange(H).to(device) / H
        enc_row1 = torch.sin(self.omega[None, :] * row[:, None])
        enc_row2 = torch.cos(self.omega[None, :] * row[:, None])
        rows = torch.cat([enc_row1.unsqueeze(1).repeat((1, W, 1)), enc_row2.unsqueeze(1).repeat((1, W, 1))], dim=-1)

        col = torch.arange(W).to(device) / W
        enc_col1 = torch.sin(self.omega[None, :] * col[:, None])
        enc_col2 = torch.cos(self.omega[None, :] * col[:, None])
        cols = torch.cat([enc_col1.unsqueeze(0).repeat((H, 1, 1)), enc_col2.unsqueeze(0).repeat((H, 1, 1))], dim=-1)

        encoding = torch.cat([rows, cols], dim=-1)
        encoding = encoding.permute(2, 0, 1).unsqueeze(0).repeat((B, 1, 1, 1))
        return encoding

    def get_encoding(self, x):
        shp1 = x.shape
        singelton = self.positional_encoding is not None \
                    and self.positional_encoding.shape[0] == shp1[0] and self.positional_encoding.shape[2:] == shp1[2:]
        if singelton:
            return self.positional_encoding
        self.positional_encoding = self.encode(x.shape)
        return self.positional_encoding

    def __init__(self, input_channels, output_channels, bilinear=True, padding='zero', full_ada=False,
                 nfreq=20, magnitude=10):
        super(PosADANet, self).__init__()
        factor = 2 if bilinear else 1
        self.omega = nn.Parameter(torch.rand(nfreq) * magnitude)
        self.omega.requires_grad = False
        self.positional_encoding = None
        self.full_ada = full_ada
        self.nfreq = nfreq
        

        
        self.padding = padding
        self.input_channels = input_channels + nfreq * 4
        self.n_classes = output_channels
        self.bilinear = bilinear
        self.channels = [512 // factor, 256 // factor, 128 // factor]
        self.inc = DoubleConv(self.input_channels, 64)
        self.down1 = Down(64, 128, padding=padding, ada=self.full_ada)
        self.down2 = Down(128, 256, padding=padding, ada=self.full_ada)
        self.down3 = Down(256, 512, padding=padding, ada=self.full_ada)
        self.down4 = Down(512, 1024 // factor, padding=padding, ada=self.full_ada)
        self.up1 = Up(1024, 512 // factor, bilinear, ada=False, padding=padding)
        self.up2 = Up(512, 256 // factor, bilinear, ada=False, padding=padding)
        self.up3 = Up(256, 128 // factor, bilinear, ada=False, padding=padding)
        self.up4 = Up(128, 64, bilinear, padding=padding, ada=False)
        self.outc = OutConv(64, output_channels, padding=padding)

    def forward(self, x):
        #w = self.style_encoder(style)
        if self.nfreq > 0:
            encoding = self.get_encoding(x)
            x = torch.cat([x, encoding], dim=1)

        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

        """
        # this part separetes the blue channel (which acts as an alpha) from the red and green
        #rg,b = torch.split(logits,[2,1],dim=1)
        #rg = F.tanh(rg) # normalize between -1 and 1
        #b = (F.tanh(b) + 1)/2 # normalize between 0 and 1
        #return torch.cat((rg,b),dim=1)
        """

