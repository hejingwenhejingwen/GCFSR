import argparse
import cv2
import glob
import numpy as np
import os

from basicsr.utils.matlab_functions import imresize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/celeba_val', help='input GT image folder')
    parser.add_argument('--output', type=str, default='datasets/celeba_val_input', help='output LR image folder')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    scales = [8, 16, 32, 64]

    # set up model
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*.jpg')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('make', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

        for scale in scales:
            out = imresize(img, 1/scale)
            # out = imresize(out, scale)
            
            out = np.clip(out, 0, 1)
            out = (out * 255.0).round().astype(np.uint8)

            cv2.imwrite(os.path.join(args.output, f'{imgname}_down{scale}.png'), out)
        

        
if __name__ == '__main__':
    main()
