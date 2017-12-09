import os
from os.path import join, exists
from PIL import Image

IMAGE_ROOT = '../img'
ORIGINAL_IMAGE_ROOT = join(IMAGE_ROOT, 'original')
BW_IMAGE_ROOT = join(IMAGE_ROOT, 'bw')


def generate_bw_images_and_list():
    """Generate corresponding black and white images for all images under img/original folder """

    for path, subdirs, files in os.walk(ORIGINAL_IMAGE_ROOT):
        for subdir in subdirs:
            if not exists(join(BW_IMAGE_ROOT, subdir)):
                os.makedirs(join(BW_IMAGE_ROOT, subdir))
        for name in files:
            if name[-3:] in ['jpg', 'png']:
                original_image = Image.open(join(path, name))
                bw_image = original_image.convert('L')
                bw_image.save(join(BW_IMAGE_ROOT, path[16:], name))
    # TODO: generate a list of filenames


if __name__ == '__main__':
    generate_bw_images_and_list()