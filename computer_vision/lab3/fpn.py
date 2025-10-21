from collections import OrderedDict
import torch.nn.functional as F
from torch import nn
from utils import ConvNormActBlock


class FeaturePyramidNetwork(nn.Module):
    def __init__(
        self,
        input_keys_list,
        input_channels_list,
        output_channels,
        pool=True,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        self.input_keys_list = input_keys_list
        self.input_channels_list = input_channels_list
        self.out_channels = output_channels
        self.pool = pool

        self.channel_projections = nn.ModuleList()
        self.blend_convs = nn.ModuleList()
        for in_channels in input_channels_list:
            self.channel_projections.append(
                ConvNormActBlock(
                    in_channels, output_channels, kernel_size=1, norm_layer=norm_layer
                )
            )
            self.blend_convs.append(
                ConvNormActBlock(
                    output_channels, output_channels, kernel_size=3, norm_layer=norm_layer
                )
            )

    def forward(self, x):
        out = dict()
        # Write a forward pass for the FPN and save the results inside the dictionary out
        # with keys "fpn2", "fpn3", "fpn4", "fpn5" and "fpn_pool".
        # Argument x is also a dictionary and contains the features
        # from the backbone labeled with keys "res2", "res3", "res4" and "res5".
        # Be careful with the order of the feature maps.
        # Index 0 in self.blend_convs and self.channel_projections corresponds
        # tp the finest level of the pyramid, i.e. operates on the features "res2"
        # and computes output features "fpn2".
        # Similarly, layers at the end of the list correspond to the coarsest level
        # of the pyramid, i.e. features "res5/fpn5".
        # Use maxpool with stride 2 and kernel size 1 applied to features "fpn5" in order
        # to compute "fpn_pool" features.
        # Use F.interpolate with mode "nearest" to upsample the features.
        # YOUR CODE HERE

        x5 = x["res5"]
        channel5 = self.channel_projections[3](x5)
        blend5 = self.blend_convs[3](channel5)

        upsample5 = F.interpolate(channel5, scale_factor=2, mode='nearest')

        x4 = x["res4"]
        channel4 = self.channel_projections[2](x4)
        sum4 = channel4 + upsample5
        blend4 = self.blend_convs[2](sum4)

        upsample4 = F.interpolate(sum4, scale_factor=2, mode='nearest')

        x3 = x["res3"]
        channel3 = self.channel_projections[1](x3)
        sum3 = channel3 + upsample4
        blend3 = self.blend_convs[1](sum3)

        upsample3 = F.interpolate(sum3, scale_factor=2, mode='nearest')

        x2 = x["res2"]
        channel2 = self.channel_projections[0](x2)
        sum2 = channel2 + upsample3
        blend2 = self.blend_convs[0](sum2)

        out["fpn2"] = blend2
        out["fpn3"] = blend3
        out["fpn4"] = blend4
        out["fpn5"] = blend5
        
        out["fpn_pool"] = F.max_pool2d(out["fpn5"], kernel_size=1, stride=2)


        # Rest of the code expects a dictionary with properly ordered keys.
        ordered_out = OrderedDict()
        for k in ["fpn2", "fpn3", "fpn4", "fpn5", "fpn_pool"]:
            ordered_out[k] = out[k]

        return ordered_out
