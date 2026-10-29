import time
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image


def apply_sobel(chunk):
    sobelx = cv2.Sobel(chunk, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(chunk, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(sobelx, sobely)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    return magnitude


def process_image(image_path, output_path, num_processes):
    try:
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"Error: Nie znaleziono pliku {image_path}")
        return

    img_array = np.array(img)
    chunks = np.array_split(img_array, num_processes, axis=0)

    start_time = time.time()

    with Pool(processes=num_processes) as pool:
        processed_chunks = pool.map(apply_sobel, chunks)

    end_time = time.time()
    print(f"Czas: {end_time - start_time:.4f} s")

    final_array = np.vstack(processed_chunks)
    final_image = Image.fromarray(final_array)
    final_image.save(output_path)


if __name__ == '__main__':
    input_image_path = '401.jpg'
    output_image_path = 'output.jpg'
    num_workers = 4

    process_image(input_image_path, output_image_path, num_workers)
