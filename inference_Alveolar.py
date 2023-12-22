import os
import numpy as np
import torch
from torch import nn
from models.ModelFactory import ModelFactory
import torchio as tio
from torch.utils.data import DataLoader
import nibabel as nib
from scipy.ndimage import zoom
from argparse import ArgumentParser
import sys


# parse arguments
parser = ArgumentParser(description="inference_AlveolarNerve", epilog='\n')

# input/outputs
parser.add_argument("--weights", help="Path to weight to be loaded")
parser.add_argument("--i", help="Image to segment.")
parser.add_argument("--o", help="Segmentation output.")
parser.add_argument("--cpu", action="store_true", help="(optional) Enforce running with CPU rather than GPU.")

# check for no arguments
if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(1)

# parse commandline
args = vars(parser.parse_args())

config = {
    "data_loader": {
        "resize_shape": [168, 280, 360],
        "patch_shape" : [80, 80, 80]
    }
    
}

def _numpy_reader(self, path=args["i"]):
        data = torch.from_numpy(nib.load(path).get_fdata()).float()
        affine = torch.from_numpy(nib.load(path).affine)
        return data, affine

image_scalar = tio.ScalarImage(args["i"], reader=_numpy_reader)

#resizing for images with different voxel spacings and densities
#resized_array = zoom(image_scalar.data.numpy(), (1,1,1,2))

rotated_image_tensor = torch.from_numpy(np.rot90(image_scalar.data.numpy(), k=1, axes=(1, 3)).copy())

#for rotating when patient is facing "wrong" way
#rotated_image_tensor = torch.from_numpy(np.rot90(rotated_image_tensor, k=1, axes=(2, 3)).copy())
#rotated_image_tensor = torch.from_numpy(np.rot90(rotated_image_tensor, k=1, axes=(2, 3)).copy())

# Create a new ScalarImage
image_scalar = tio.ScalarImage(tensor=rotated_image_tensor, affine=image_scalar.affine)

transforms = []

transforms.append(tio.transforms.Clamp(out_min = 0, out_max = 2100))
transforms.append(tio.transforms.RescaleIntensity())

preprocessing = tio.Compose(transforms)
image_preprocessed = preprocessing(image_scalar)

subject = tio.Subject(image=image_preprocessed)

emb_shape = [dim // 8 for dim in config["data_loader"]["patch_shape"]]
model = ModelFactory("PosPadUNet3D", 1, 1, emb_shape).get()
model = nn.DataParallel(model)




if(args['cpu'] or torch.cuda.is_available() is False):      
    model.load_state_dict(torch.load(args["weights"], map_location=torch.device('cpu'))['state_dict'])
    model = model.module.to('cpu')
    print("CPU")
else:
    model.load_state_dict(torch.load(args["weights"])['state_dict'])
    print("CUDA")
    

model.eval()

with torch.no_grad():

    crop_or_pad_transform = tio.CropOrPad(config["data_loader"]["resize_shape"], padding_mode=0)

    sampler = tio.inference.GridSampler(
            subject,
            config["data_loader"]["patch_shape"],
            patch_overlap=0,
    )
    loader = DataLoader(sampler, batch_size=2)
    aggregator = tio.inference.GridAggregator(sampler, overlap_mode='average')

    for j, patch in enumerate(loader):
        if(torch.cuda.is_available and not args['cpu']):     
            images = patch['image'][tio.DATA].float().cuda()  # BS, 3, Z, H, W
            emb_codes = patch[tio.LOCATION].float().cuda()
        else:
            images = patch['image'][tio.DATA].float()  # BS, 3, Z, H, W
            emb_codes = patch[tio.LOCATION].float()

        output = model(images, emb_codes)  # BS, Classes, Z, H, W
        aggregator.add_batch(output, patch[tio.LOCATION])

    output = aggregator.get_output_tensor()
    # output = tio.CropOrPad(original_shape, padding_mode=0)(output)
    output = output.squeeze(0)
    output = (output > 0.5).int()
    output = output.detach().cpu().numpy()  # BS, Z, H, W
    output = np.rot90(output, k=-1, axes=(0, 2)).copy()

    print(image_scalar.affine)
    nifti_img = nib.Nifti1Image(output, image_scalar.affine)

    # Save the NIfTI image
    nib.save(nifti_img, args['o'])