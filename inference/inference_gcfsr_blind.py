import argparse
import cv2
import math
import glob
import numpy as np
import os
import torch
from torchvision.transforms.functional import normalize
from torchvision import utils

from basicsr.data import degradations as degradations
from basicsr.archs.gcfsr_arch import GCFSR_blind
from basicsr.utils.matlab_functions import imresize


def generate(args, img, g_ema, device, imgname):

    with torch.no_grad():
        output, _ = g_ema(img)

        output = output.data.squeeze().float().cpu().clamp_(-1, 1).numpy()
        output = (output + 1) / 2
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(args.output, f'{imgname}.png'), output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'experiments/pretrained_models/gcfsr_blind_512.pth'  # noqa: E501
    )
    parser.add_argument('--input', type=str, default='', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/tmp', help='output folder')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output, exist_ok=True)

    # set up model
    model = GCFSR_blind(512)
    model.load_state_dict(torch.load(args.model_path), strict=True)
    model.eval()
    model = model.to(device)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        h, w, _ = img.shape

        ###### numpy to tensor, BGR to RGB
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        ###### clamp
        img = torch.clamp((img * 255.0).round(), 0, 255) / 255.
        ###### normalization
        normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
        img = img.unsqueeze(0).to(device)

        ###### setting conditions
        generate(args, img, model, device, imgname)
        
if __name__ == '__main__':
    main()
