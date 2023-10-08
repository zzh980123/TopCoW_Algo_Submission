from monai.networks.nets import *
from .mednextv1.create_mednext_v1 import *
from .UXNet_3D.network_backbone import *


def model_factory(model_name: str, device, args):
    if model_name == 'segresnet':
        model = SegResNet(
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            init_filters=16,
            in_channels=args.in_channels,
            out_channels=args.output_classes,
            dropout_prob=0.2,
        ).to(device)

    if model_name == 'swinunetr':
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=args.in_channels,
            out_channels=args.output_classes,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
        ).to(device)

    if model_name == 'unet':
        model = UNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.output_classes,
            channels=(4, 8, 16, 32, 64),
            strides=(2, 2, 2, 2),
        ).to(device)

    if model_name == 'mednext3':
        model = create_mednext_v1(
            num_input_channels=args.in_channels,
            num_classes=args.output_classes,
            model_id=args.model_id,
            kernel_size=3,
            deep_supervision=args.deep_supervision,
        ).to(device)

    if model_name == 'mednext5':
        model = create_mednext_v1(
            num_input_channels=args.in_channels,
            num_classes=args.output_classes,
            model_id=args.model_id,
            kernel_size=5,
            deep_supervision=args.deep_supervision,
        ).to(device)

    if model_name == 'uxnet':
        model = UXNET(
            in_chans=args.in_channels,
            out_chans=args.output_classes,
            feat_size=[36, 72, 144, 288],
            hidden_size=768,
        ).to(device)

    return model

