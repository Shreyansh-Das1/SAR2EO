import torch
import torch.nn as nn

class UNetSkipConnectionBlock(nn.Module):
    """Defines a helper block for building recursive U-Net skip-connections."""
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False):
        super(UNetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # Maintain spatial registration across modalities via structural skip links
            return torch.cat([x, self.model(x)], 1)


class UNetGenerator(nn.Module):
    """U-Net Generator Architecture matching paired structural constraints."""
    def __init__(self, input_nc=1, output_nc=3, num_filters=64):
        super(UNetGenerator, self).__init__()
        # Construct U-Net architecture from the inside out
        unet_block = UNetSkipConnectionBlock(num_filters * 8, num_filters * 8, submodule=None, innermost=True)
        for _ in range(3):
            unet_block = UNetSkipConnectionBlock(num_filters * 8, num_filters * 8, submodule=unet_block)
        unet_block = UNetSkipConnectionBlock(num_filters * 4, num_filters * 8, submodule=unet_block)
        unet_block = UNetSkipConnectionBlock(num_filters * 2, num_filters * 4, submodule=unet_block)
        unet_block = UNetSkipConnectionBlock(num_filters, num_filters * 2, submodule=unet_block)
        self.model = UNetSkipConnectionBlock(output_nc, num_filters, input_nc=input_nc, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    """PatchGAN architecture classifying 70x70 image patches as real or fake."""
    def __init__(self, input_nc=4, num_filters=64): # Paired inputs: SAR (1) + EO (3) = 4 channels
        super(PatchGANDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, num_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=1) # Patch validity mapping
        )

    def forward(self, x):
        return self.model(x)
