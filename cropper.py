#!/usr/bin/env python

""" Code to rotate and crop a sports card image. I found the function here:
https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python/48553593#48553593
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
import glob
import argparse

class ImageProcessor:
    def __init__(self, source_dir='.', backup_dir='backup', overwrite=False, margin=0, upper_threshold=235):
        self.source_dir = Path(source_dir)
        self.backup_dir = self.source_dir / backup_dir
        self.backup_dir.mkdir(exist_ok=True)
        self.overwrite = overwrite
        self.margin = margin
        self.upper_threshold = upper_threshold
        self.debug = False

    def get_sub_image(self, rect, image):
        center, size, theta = rect
        center, size = tuple(map(int, center)), tuple(map(int, size))
        # Add margin to size
        size = (size[0] + 2 * self.margin, size[1] + 2 * self.margin)
        
        # Tries to handle rotating clock-wise
        if theta > 45:
            theta = -(90-theta)
            size = (size[1], size[0])
        
        M = cv2.getRotationMatrix2D(center, theta, 1)
        dst = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        if self.debug:
            cv2.imshow('Rotated', dst)
            cv2.waitKey(0)
        out = cv2.getRectSubPix(dst, size, center)
        return out

    def backup_image(self, image_file):
        shutil.copy(image_file, self.backup_dir / image_file.name)

    def debug_contour(self, image, contours):
        # Create a blank image to draw the contours on
        contour_image = image.copy()
        #contour_image = np.zeros_like(image)  # blank image

        # Draw the contours on the image
        cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)
        rect = cv2.minAreaRect(contours)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(contour_image, [box], -1, (0, 0, 255), 2)

        # Show the contour image
        cv2.imshow('Contours', contour_image)
        cv2.waitKey(0)

    def process_image(self, image_file):
        img = cv2.imread(str(image_file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lower = (0, 0, 0)
        upper = (self.upper_threshold, self.upper_threshold, self.upper_threshold)
        thresh = cv2.inRange(img, lower, upper)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)
        if self.debug:
            self.debug_contour(img, big_contour)
        rect = cv2.minAreaRect(big_contour)
        out = self.get_sub_image(rect, img)

        if self.overwrite:
            output_file = image_file
        else:
            output_file = self.source_dir / (image_file.stem + '_out.jpg')

        cv2.imwrite(str(output_file), out)

    def process_images(self):
        image_files = list(self.source_dir.glob('*.jpg')) + list(self.source_dir.glob('*.JPG'))
        for image_file in image_files:
            if not image_file.stem.endswith('_out'):
                print(f'Rotating and cropping: {image_file}')
                self.backup_image(image_file)
                self.process_image(image_file)

def main():
    parser = argparse.ArgumentParser(description='Rotate and crop sports card images.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input files (wildcard supported)')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite original image file.')
    parser.add_argument('-m', '--margin', type=int, default=0, help='Margin to add around the cropped image.')
    parser.add_argument('-u', '--upper_threshold', type=int, default=235, help='Upper bound for threshold (default 235).')

    args = parser.parse_args()
    input_files = glob.glob(args.input)

    processor = ImageProcessor(overwrite=args.overwrite, margin=args.margin, upper_threshold=args.upper_threshold)

    if args.input:
        for file in input_files:
            image_file = Path(file)
            if image_file.exists() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                print(f'Cropping: {image_file}')
                processor.backup_image(image_file)
                processor.process_image(image_file)
            else:
                print(f"Error: File {args.input} not found or unsupported format.")
    else:
        processor.process_images()

if __name__ == "__main__":
    main()
