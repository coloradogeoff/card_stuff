#! /usr/bin/env python

import sys
import os
import re
import glob
import time
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

def average_perimeter_color(images):
    all_perims = [get_perimeter_pixels(img) for img in images]
    combined = np.vstack(all_perims)
    return tuple(combined.mean(axis=0).astype(np.uint8))

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

        for i in range(2):
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

        fill_color = average_perimeter_color(images)

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
    return output_file


def touch_files_in_order(card_files, merged_file):
    # Ensure deterministic date ordering:
    # merged file is newest, then card_...0001.jpg, card_...0002.jpg, etc.
    base_time = time.time()
    os.utime(merged_file, (base_time, base_time))
    for i, path in enumerate(card_files, start=1):
        ts = base_time - i
        os.utime(path, (ts, ts))

def main():
    # Check if -e is provided. If yes, use even numbers; otherwise, default to odd.
    use_even = False
    args = sys.argv[1:]
    if '-e' in args:
        use_even = True
        args.remove('-e')

    if not args:
        args = ['card*']
    
    # Expand any wildcard arguments into a complete list of files.
    files = []
    for arg in args:
        if '*' in arg or '?' in arg:
            files.extend(glob.glob(arg))
        else:
            files.append(arg)
    
    pattern = re.compile(r'(\d+)(?=\.jpg$)', re.IGNORECASE)
    file_nums = {f: int(m.group(1)) for f in files if (m := pattern.search(f))}

    parity = 0 if use_even else 1
    filtered_files = sorted(
        [f for f, n in file_nums.items() if n % 2 == parity],
        key=file_nums.get,
    )
    ordered_touch_files = sorted(file_nums, key=file_nums.get)

    print("Merging files:", filtered_files)
    merged_file = merge_images(filtered_files)
    touch_files_in_order(ordered_touch_files, merged_file)

if __name__ == "__main__":
    main()
