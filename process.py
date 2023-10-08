"""
The most important file for Grand-Challenge Algorithm submission is this process.py.
This is the file where you will extend our base algorithm class,
and modify the subclass of MyCoWSegAlgorithm for your awesome algorithm :)
Simply update the TODO in this file.

NOTE: remember to COPY your required files in your Dockerfile
COPY --chown=user:user <somefile> /opt/app/
"""

import SimpleITK as sitk
import argparse
import ants
import torch.nn.functional
import time
from base_algorithm import TASK, TRACK, BaseAlgorithm
from models.model_selector import *
import monai
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    ScaleIntensityRangePercentiles,
    NormalizeIntensity,
    EnsureChannelFirst,
    EnsureType,
    MapTransform,
    Invert,
    LoadImaged,
    Orientation,
    CastToType,
    Activations,
    Activations,
    AsDiscrete,
    Resize,
    SpatialPad,
)
import numpy as np
from skimage import morphology, measure
from skimage.morphology import closing, square, ball
from skimage.segmentation import clear_border

#######################################################################################
# TODO: First, choose your track and task!
# track is either TRACK.CT or TRACK.MR
# task is either TASK.BINARY_SEGMENTATION or TASK.MULTICLASS_SEGMENTATION
track = TRACK.MR
task = TASK.MULTICLASS_SEGMENTATION


# END OF TODO
#######################################################################################


class MyCoWSegAlgorithm(BaseAlgorithm):
    """
    Your algorithm goes here.
    Simply update the TODO in this file.
    """

    def __init__(self):
        super().__init__(
            track=track,
            task=task,
        )
        self.bin_patch_size = [192, 192, 64]
        self.patch_size = [192, 192, 64]
        self.infer_transform = Compose(
            [
                # EnsureChannelFirst(),
                EnsureType(),
                ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_min=0, b_max=1),
                NormalizeIntensity(nonzero=True, channel_wise=True),
                # CastToType(torch.float32)
            ]
        )
        self.post_bin = Compose(
            [
                Activations(sigmoid=True),
                AsDiscrete(argmax=False, logit_thresh=0.4)
            ]
        )
        self.post_transform = Compose(
            [
                Activations(softmax=True),
                AsDiscrete(argmax=True),
            ]
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bin_model_0 = create_mednext_v1(
            num_input_channels=1,
            num_classes=1,
            model_id='L',
            kernel_size=3,
            deep_supervision=False
        ).to(self.device)

        bin_model_1 = UXNET(
            in_chans=1,
            out_chans=1,
            feat_size=[36, 72, 144, 288],
            hidden_size=768,
        ).to(self.device)

        mul_model_0 = create_mednext_v1(
            num_input_channels=2,
            num_classes=14,
            model_id='M',
            kernel_size=5,
            deep_supervision=False
        ).to(self.device)

        mul_model_1 = UXNET(
            in_chans=2,
            out_chans=14,
            feat_size=[36, 72, 144, 288],
            hidden_size=768,
        ).to(self.device)

        bin_model_0_weight_path = [
            'workdir/mr/bin/bin_mednext3_L_192_fold0.pth',
            'workdir/mr/bin/bin_mednext3_L_192_fold1.pth',
            'workdir/mr/bin/bin_mednext3_L_192_fold2.pth',
            'workdir/mr/bin/bin_mednext3_L_192_fold3.pth',
            'workdir/mr/bin/bin_mednext3_L_192_fold4.pth',
        ]
        bin_model_1_weight_path = [
            'workdir/mr/bin/bin_uxnet_192_fold0.pth',
            'workdir/mr/bin/bin_uxnet_192_fold1.pth',
            'workdir/mr/bin/bin_uxnet_192_fold2.pth',
            'workdir/mr/bin/bin_uxnet_192_fold3.pth',
            'workdir/mr/bin/bin_uxnet_192_fold4.pth',
        ]
        mul_model_0_weight_path = [
            'workdir/mr/mul/mul_mednext5_drop_192_fold0.pth',
            'workdir/mr/mul/mul_mednext5_drop_192_fold1.pth',
            'workdir/mr/mul/mul_mednext5_drop_192_fold2.pth',
            'workdir/mr/mul/mul_mednext5_drop_192_fold3.pth',
            'workdir/mr/mul/mul_mednext5_drop_192_fold4.pth',
        ]
        mul_model_1_weight_path = [
            'workdir/mr/mul/mul_uxnet_drop_192_fold0.pth',
            'workdir/mr/mul/mul_uxnet_drop_192_fold1.pth',
            'workdir/mr/mul/mul_uxnet_drop_192_fold2.pth',
            'workdir/mr/mul/mul_uxnet_drop_192_fold3.pth',
            'workdir/mr/mul/mul_uxnet_drop_192_fold4.pth',
        ]

        # load bin net
        self.bin_net_list = []
        for model_weight in bin_model_0_weight_path:
            bin_checkpoint = torch.load(model_weight, map_location=torch.device(self.device))
            bin_model_0.load_state_dict(bin_checkpoint)
            bin_model_0.eval()
            self.bin_net_list.append(bin_model_0)

        for model_weight in bin_model_1_weight_path:
            bin_checkpoint = torch.load(model_weight, map_location=torch.device(self.device))
            bin_model_1.load_state_dict(bin_checkpoint)
            bin_model_1.eval()
            self.bin_net_list.append(bin_model_1)

        # load mul net
        self.mul_net_list = []
        for model_weight in mul_model_0_weight_path:
            mul_mednext_checkpoint = torch.load(model_weight, map_location=torch.device(self.device))
            mul_model_0.load_state_dict(mul_mednext_checkpoint)
            mul_model_0.eval()
            self.mul_net_list.append(mul_model_0)

        for model_weight in mul_model_1_weight_path:
            mul_uxnet_checkpoint = torch.load(model_weight, map_location=torch.device(self.device))
            mul_model_1.load_state_dict(mul_uxnet_checkpoint)
            mul_model_1.eval()
            self.mul_net_list.append(mul_model_1)

        self.template_image_path = r'workdir/mr/topcow_mr_086_rai.nii.gz'
        self.template_roi_path = r'workdir/mr/topcow_mr_086_cube_rai.nii.gz'

    def ants_2_itk(self, image):
        imageITK = sitk.GetImageFromArray(image.numpy().T)
        imageITK.SetOrigin(image.origin)
        imageITK.SetSpacing(image.spacing)
        imageITK.SetDirection(image.direction.reshape(9))
        return imageITK

    def itk_2_ants(self, image):
        image_ants = ants.from_numpy(sitk.GetArrayFromImage(image).T,
                                     origin=image.GetOrigin(),
                                     spacing=image.GetSpacing(),
                                     direction=np.array(image.GetDirection()).reshape(3, 3))
        return image_ants

    def load_net(self, model, model_weight_path, device, net):
        for model_weight in model_weight_path:
            print(f"Loading {model_weight}...")
            checkpoint = torch.load(model_weight, map_location=torch.device(device))
            model.load_state_dict(checkpoint)
            model.eval()
            net.append(model)
        return net

    def post_process_bin(self, pred_bin, hole_size, min_size):
        pred_bin = torch.squeeze(torch.squeeze(pred_bin, 0), 0)
        pred_bin = pred_bin.cpu().numpy().astype(np.uint8)
        pred_bin = closing(pred_bin, ball(1))
        pred_bin = morphology.remove_small_holes(pred_bin, hole_size).astype(np.uint8)
        pred_bin = morphology.remove_small_objects(pred_bin, min_size).astype(np.uint8)

        pred_bin = torch.from_numpy(pred_bin).to(self.device)
        pred_bin = torch.unsqueeze(torch.unsqueeze(pred_bin, 0), 0)
        return pred_bin

    def post_process_labels(self, pred_mul, min_size, hole_size):
        unique_labels = np.unique(pred_mul)
        processed_labels = np.zeros_like(pred_mul)

        for label in unique_labels:
            mask = (pred_mul == label)

            mask = morphology.remove_small_holes(mask, hole_size)

            # mask = closing(mask, ball(1))

            labeled_image = measure.label(mask)

            properties = measure.regionprops(labeled_image)

            max_area = 0
            max_label = 0
            for prop in properties:
                if prop.area >= min_size and prop.area > max_area:
                    max_area = prop.area
                    max_label = prop.label

            if max_label != 0:
                processed_labels[labeled_image == max_label] = label

        return processed_labels

    def predict(self, *, image_ct: sitk.Image, image_mr: sitk.Image) -> np.array:
        """
        Inputs will be a pair of CT and MR .mha SimpleITK.Images
        Output is supposed to be a numpy array in (x,y,z)
        """

        #######################################################################################
        # TODO: place your own prediction algorithm here
        # You are free to remove everything! Just return to us an npy in (x,y,z)
        # NOTE: If you extract the array from SimpleITK, note that
        #              SimpleITK npy array axis order is (z,y,x).
        #              Then you might have to transpose this to (x,y,z)
        #              (see below for an example).
        #######################################################################################
        if track == TRACK.MR:
            print("-> main_input is from TRACK.MR")
            main_input = image_mr
        else:
            print("-> main_input is from TRACK.CT")
            main_input = image_ct
        print("===============start==============")
        t0 = time.time()
        # 1. load template and set direction
        main_input = sitk.Cast(main_input, sitk.sitkFloat32)
        copied_main_input = sitk.Image(main_input)
        rai_direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        move_img = sitk.ReadImage(self.template_image_path)
        move_label = sitk.ReadImage(self.template_roi_path)
        copied_main_input.SetDirection(rai_direction)
        # move_img.SetDirection(rai_direction)
        # move_label.SetDirection(rai_direction)

        # 2. registration and get roi bbox
        ants_fix_img = self.itk_2_ants(main_input)
        ants_move_img = self.itk_2_ants(move_img)
        ants_move_label = self.itk_2_ants(move_label)

        reg_out = ants.registration(ants_fix_img, ants_move_img, type_of_transform="Affine")
        reg_label = ants.apply_transforms(ants_fix_img, ants_move_label, transformlist=reg_out['fwdtransforms'],
                                          interpolator='nearestNeighbor')
        roi_array = reg_label.numpy()
        array_mr = sitk.GetArrayFromImage(main_input).transpose((2, 1, 0))
        if not np.any(roi_array):
            x_min = int(array_mr.shape[0] * 0.2)
            x_max = int(array_mr.shape[0] * 0.7)
            y_min = int(array_mr.shape[1] * 0.2)
            y_max = int(array_mr.shape[1] * 0.7)
            z_min = int(array_mr.shape[2] * 0.2)
            z_max = int(array_mr.shape[2] * 0.8)
        else:
            coordinates = np.argwhere(roi_array == 1)

            x_min = coordinates[:, 0].min()
            x_max = coordinates[:, 0].max()
            y_min = coordinates[:, 1].min()
            y_max = coordinates[:, 1].max()
            z_min = coordinates[:, 2].min()
            z_max = coordinates[:, 2].max()
        print(f'{x_min=}, {x_max=}, {y_min=}, {y_max=}, {z_min=}, {z_max=}')
        pre_cost = time.time() - t0
        print(f'{pre_cost=}')

        input_array = sitk.GetArrayFromImage(main_input).transpose((2, 1, 0))
        input_array = np.expand_dims(input_array, 0)

        with torch.no_grad():
            image = self.infer_transform(input_array).to(self.device)
            _, w, h, d = image.shape
            image = torch.unsqueeze(image, 0)
            seg_data = image[..., x_min:x_max, y_min:y_max, z_min:z_max]
            sw_batch_size = 1
            # seg_shape = seg_data.shape
            #
            # 3. get bin pred
            bin_out = None
            for net in self.bin_net_list:
                bin_pred = sliding_window_inference(
                    seg_data,
                    self.bin_patch_size,
                    sw_batch_size,
                    net,
                    device=self.device,
                    mode="gaussian",
                    overlap=0.5,
                )
                bin_pred = self.post_bin(torch.squeeze(bin_pred))
                if bin_out is None:
                    bin_out = bin_pred
                else:
                    bin_out[bin_pred == 1] = 1
            bin_out = torch.unsqueeze(torch.unsqueeze(bin_out, 0), 0)
            # bin_pred /= len(self.bin_net_list)
            print(f'{len(self.bin_net_list)=}')
            # bin_out = torch.unsqueeze(torch.unsqueeze(self.post_bin(torch.squeeze(bin_pred)), 0), 0)
            # bin_out = self.post_process_bin(bin_out, hole_size=512, min_size=1024)

            # 4. get mul-model input
            mul_seg_data = torch.cat((seg_data, bin_out), dim=1)
            # mul_seg_data = self.gen_coord(mul_seg_data).to(self.device)
            print(f'{mul_seg_data.shape=}')

            # 4. get mul pred
            mul_pred = 0
            for net in self.mul_net_list:
                mul_pred += sliding_window_inference(
                    mul_seg_data,
                    self.patch_size,
                    # [w, h, d],
                    sw_batch_size,
                    net,
                    device=self.device,
                    # device=torch.device('cpu'),
                    mode="gaussian",
                    overlap=0.75,
                )
            mul_pred /= len(self.mul_net_list)
            print(f'{len(self.mul_net_list)=}')
            post_pred = self.post_transform(torch.squeeze(mul_pred)).cpu().astype(np.int8)
            # post_pred = Resize(spatial_size=seg_shape)(post_pred)
            post_pred[post_pred == 13] = 15
            out_pred = np.zeros_like(image[0][0])
            out_pred[x_min:x_max, y_min:y_max, z_min:z_max] = post_pred[0]
        print("pred_array.shape = ", out_pred.shape)
        # The output np.array needs to have the same shape as track modality input
        # print(f"main_input.GetSize() = {main_input.GetSize()}")

        # END OF TODO
        #######################################################################################

        out_pred = self.post_process_labels(out_pred, 128, 1024)
        total_cost = time.time() - t0
        print(f'{total_cost=}')
        # out_pred = remove_small_objects(out_pred.astype(np.int8), 1024)
        # return prediction array
        return out_pred


if __name__ == "__main__":
    # NOTE: running locally ($ python3 process.py) has advantage of faster debugging
    # but please ensure the docker environment also works before submitting
    MyCoWSegAlgorithm().process()
    cowsay_msg = """\n
  ____________________________________
< MyCoWSegAlgorithm().process()  Done! >
  ------------------------------------
         \   ^__^ 
          \  (oo)\_______
             (__)\       )\/\\
                 ||----w |
                 ||     ||
    """
    print(cowsay_msg)
