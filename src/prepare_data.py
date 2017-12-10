import os
from os.path import join, exists
from PIL import Image

IMAGE_ROOT = '../img'
ORIGINAL_IMAGE_ROOT = join(IMAGE_ROOT, 'original')
BW_IMAGE_ROOT = join(IMAGE_ROOT, 'bw')
AUTO_IMAGE_ROOT = join(IMAGE_ROOT, 'auto')
RESULT_IMAGE_ROOT = join(IMAGE_ROOT, 'result')
SEGMENT_IMAGE_ROOT = join(IMAGE_ROOT, 'segment')
USER_IMAGE_ROOT = join(IMAGE_ROOT, 'user')

def generate_bw_images_and_list():
    """Generate corresponding black and white images for all images under img/original folder """

    img_urls = []
    for path, subdirs, files in os.walk(ORIGINAL_IMAGE_ROOT):
        for subdir in subdirs:
            if not exists(join(BW_IMAGE_ROOT, subdir)):
                os.makedirs(join(BW_IMAGE_ROOT, subdir))
            if not exists(join(AUTO_IMAGE_ROOT, subdir)):
                os.makedirs(join(AUTO_IMAGE_ROOT, subdir))
            if not exists(join(RESULT_IMAGE_ROOT, subdir)):
                os.makedirs(join(RESULT_IMAGE_ROOT, subdir))
            if not exists(join(SEGMENT_IMAGE_ROOT, subdir)):
                os.makedirs(join(SEGMENT_IMAGE_ROOT, subdir))
            if not exists(join(USER_IMAGE_ROOT, subdir)):
                os.makedirs(join(USER_IMAGE_ROOT, subdir))
        for name in files:
            if name[-3:] in ['jpg', 'png']:
                img_urls.append(join(path[16:], name))
                original_image = Image.open(join(path, name))
                bw_image = original_image.convert('L')
                bw_image.save(join(BW_IMAGE_ROOT, path[16:], name))
    return img_urls

if __name__ == '__main__':
    generate_bw_images_and_list()
