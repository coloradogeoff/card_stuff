#! /usr/bin/env python

import sys
import os
import re
import numpy as np
from PIL import Image

def get_next_filename(prefix="merged", extension=".jpg"):
    i = 1
    while True:
        filename = f"{prefix}{i:03d}{extension}"
        if not os.path.exists(filename):
            return filename
        i += 1

def get_perimeter_pixels(img):
    arr = np.array(img)
    top = arr[0, :, :]
    bottom = arr[-1, :, :]
    left = arr[:, 0, :]
    right = arr[:, -1, :]
    perimeter = np.vstack([top, bottom, left, right])
    return perimeter

def create_average_fill_image(size, images):
    all_perims = [get_perimeter_pixels(img) for img in images]
    combined = np.vstack(all_perims)
    avg_color = tuple(combined.mean(axis=0).astype(np.uint8))
    return Image.new('RGB', size, avg_color)

def merge_images(image_files, output_file=None):
    n = len(image_files)
    images = [Image.open(f) for f in image_files]
    width, height = images[0].size

    if n == 2:
        cols_top = 2
        rows = 1
        merged_width = cols_top * width
        merged_height = rows * height
        merged_image = Image.new("RGB", (merged_width, merged_height))

        # Paste top 2 images
        for i in [0, 1]:
            merged_image.paste(images[i], (i * width, 0))

    elif n == 3:
        cols_top = 3
        rows = 1

        merged_width = cols_top * width
        merged_height = rows * height
        merged_image = Image.new("RGB", (merged_width, merged_height))

        # Paste top 3 images
        for i in range(3):
            merged_image.paste(images[i], (i * width, 0))

    # Special handling for exactly 5 images
    elif n == 5:
        cols_top = 3
        rows = 2

        # Create fill image (uniform background color)
        fill_img = create_average_fill_image((width, height), images)

        # Get the single color from fill_img (any pixel will have the same value)
        fill_color = fill_img.getpixel((0, 0))

        merged_width = cols_top * width
        merged_height = rows * height
        # Create merged_image with the background set to fill_color
        merged_image = Image.new("RGB", (merged_width, merged_height), color=fill_color)

        # Paste top 3 images
        for i in range(3):
            merged_image.paste(images[i], (i * width, 0))

        # Paste bottom 2 images
        for i in range(2):
            merged_image.paste(images[i + 3], (i * width + width//2, height))

    else:
        # Fallback to previous logic for even number of images
        cols = round(n / 2)
        rows = 2
        merged_width = cols * width
        merged_height = rows * height
        merged_image = Image.new("RGB", (merged_width, merged_height))

        for index, img in enumerate(images):
            row = index // cols
            col = index % cols
            merged_image.paste(img, (col * width, row * height))

    if output_file is None:
        output_file = get_next_filename()

    merged_image.save(output_file)
    print(f"Merged image saved as {output_file}")

def main():
    # Check if -e is provided. If yes, use even numbers; otherwise, default to odd.
    use_even = False
    args = sys.argv[1:]
    if '-e' in args:
        use_even = True
        args.remove('-e')
    
    # Expand any wildcard arguments into a complete list of files.
    files = []
    for arg in args:
        if '*' in arg or '?' in arg:
            files.extend(glob.glob(arg))
        else:
            files.append(arg)
    
    # Use a regular expression to extract the sequence number before the .jpg extension.
    # This assumes your files are named like: card_20250310_0001.jpg
    pattern = re.compile(r'(\d+)(?=\.jpg$)', re.IGNORECASE)
    filtered_files = []
    for file in files:
        match = pattern.search(file)
        if match:
            num = int(match.group(1))
            if use_even:
                if num % 2 == 0:
                    filtered_files.append(file)
            else:
                if num % 2 == 1:
                    filtered_files.append(file)
        else:
            # Optionally handle files that do not match the expected pattern.
            pass

    # Sort the files by the extracted number to ensure the proper order.
    def sort_key(filename):
        m = pattern.search(filename)
        return int(m.group(1)) if m else 0

    filtered_files.sort(key=sort_key)

    # Continue with your merging process using filtered_files
    print("Merging files:", filtered_files)
    merge_images(filtered_files)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_images.py image1.jpg image2.jpg ...")
    else:
        main()