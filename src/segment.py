import scipy.misc
import scipy.signal
import numpy as np
import random
from collections import deque
from PIL import Image
from skimage import io, color
from os.path import join

class ImageSegment(object):
    def __init__(self, image_filename, threshold=100, verbose=False):
        self.image = scipy.misc.imread(image_filename, mode='L')
        # self.image = Image.open(image_filename)
        self.shape = self.image.shape
        self.label = np.zeros(self.shape, dtype=np.int32)
        self.verbose = verbose
        unlabeled = set((row, col) for row in range(self.shape[0]) for col in range(self.shape[1]))
        self.segmentation = [unlabeled]
        self.gradient = np.zeros(self.shape)
        self.threshold = threshold

    def check_rep(self):
        for label in range(len(self.segmentation)):
            for pixel in self.segmentation[label]:
                if label == 1:
                    print('checking pixel %s of label %d' % (pixel, label))
                assert type(pixel) is tuple
                assert len(pixel) == 2

    def calculate_gradient(self):
        kernel_x = np.array([[+1, 0, -1],
                             [+2, 0, -2],
                             [+1, 0, -1]])
        kernel_y = np.array([[+1, +2, +1],
                             [ 0,  0,  0],
                             [-1, -2, -1]])
        gradient_x = scipy.signal.convolve2d(self.image, kernel_x, mode='same', boundary='symm')
        gradient_y = scipy.signal.convolve2d(self.image, kernel_y, mode='same', boundary='symm')
        self.gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    def segment_high_gradient(self):
        self.segmentation.append(set())
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                pixel = (row, col)
                if self.gradient[pixel] >= self.threshold:
                    self.label[pixel] = 1
                    self.segmentation[0].remove(pixel)
                    self.segmentation[1].add(pixel)

    def random_unlabeled_pixel(self):
        return random.sample(self.segmentation[0], 1)[0]

    def adjacent_unlabeled_pixels(self, pixel, mode='cross'):
        assert type(pixel) is tuple
        row = pixel[0]
        col = pixel[1]
        if mode == 'cross':
            pixels = [(row, col), (row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]
            return [p for p in pixels if p in self.segmentation[0]]
        elif mode == '3x3':
            rows = range(pixel[0] - 1, pixel[0] + 2)
            cols = range(pixel[1] - 1, pixel[1] + 2)
            return [(row, col) for row in rows for col in cols if (row, col) in self.segmentation[0]]

    def close_enough(self, old_pixel, new_pixel):
        return True
        return (self.gradient[new_pixel] < self.threshold) == (self.gradient[old_pixel] < self.threshold)
        # return abs(self.image[new_pixel] - self.image[old_pixel]) < 5

    def label_pixel(self, pixel, old_pixel=None):
        assert type(pixel) is tuple
        self.segmentation[0].remove(pixel)
        if old_pixel is None:
            self.segmentation.append(set())
            label = len(self.segmentation) - 1
            self.segmentation[label].add(pixel)
            self.label[pixel] = len(self.segmentation) - 1
            if self.verbose:
                print('Add a new label %d for pixel %s' % (self.label[pixel], pixel))
        else:
            assert type(old_pixel) is tuple
            label = self.label[old_pixel]
            self.segmentation[label].add(pixel)
            self.label[pixel] = label
            if self.verbose:
                print('Add pixel pixel %s to existing label %d' % (pixel, label))

    def segment_one_region(self):
        if self.verbose:
            print('---------------------Begin segmenting one new region')
        seed = self.random_unlabeled_pixel()
        if self.verbose:
            print('Seed = %s' % (seed,))
        newly_labeled = deque()
        newly_labeled.append(seed)
        self.label_pixel(seed)
        while len(newly_labeled) > 0:
            pixel = newly_labeled.popleft()
            if self.verbose:
                print('Got pixel %s from queue' % (pixel,))
            for adjacent_pixel in self.adjacent_unlabeled_pixels(pixel):
                if self.close_enough(adjacent_pixel, pixel):
                    self.label_pixel(adjacent_pixel, old_pixel=pixel)
                    newly_labeled.append(adjacent_pixel)
        # self.check_rep()

    def segment(self):
        if self.verbose:
            print('Begin segmenting')
        self.segment_high_gradient()
        while len(self.segmentation[0]) != 0:
            if self.verbose:
                print('There are %d unlabeled pixels left' % len(self.segmentation[0]))
            self.segment_one_region()
        if self.verbose:
            print('Finished segmenting')

    def random_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def save_segmentation(self, output_filename):
        original_image = Image.fromarray(self.image, 'L')
        gradient_image = Image.fromarray(self.gradient)
        segment_image = Image.new("RGB", (self.shape[1], self.shape[0]))
        pixels = segment_image.load()
        for label in range(len(self.segmentation)):
            color = self.random_color()
            for pixel in self.segmentation[label]:
                # print('Got pixel %s with label %d' % (pixel, label))
                pixels[pixel[1], pixel[0]] = color
        comparison_image = Image.new("RGB", (self.shape[1] * 3, self.shape[0]))
        comparison_image.paste(original_image, (0, 0))
        comparison_image.paste(gradient_image, (self.shape[1], 0))
        comparison_image.paste(segment_image, (self.shape[1] * 2, 0))
        comparison_image.save(output_filename)

    def get_sample_points(self):
        sample_points = []
        # Sample one point from all labeled sets except label 0 (unlabeled) and label 1 (edge)
        for label in range(2, len(self.segmentation)):
            # Only sample from region that's not too small
            if len(self.segmentation[label]) > self.image.size / 1000:
                sample_points.append(random.sample(self.segmentation[label], 1)[0])
        return sample_points

    def run(self):
        self.calculate_gradient()
        self.segment()
        return self.get_sample_points()
        # self.show_segment()


def get_color_points(filename):
    auto_filename = join('../img/auto', filename)
    bw_filename = join('../img/bw', filename)
    image_segment = ImageSegment(bw_filename, verbose=False)
    sample_points = image_segment.run()
    image_segment.save_segmentation(join('../img/segment', filename))

    rgb = io.imread(auto_filename)
    lab = color.rgb2lab(rgb)
    return [((pixel[0]/image_segment.shape[0], pixel[1]/image_segment.shape[1]), lab[pixel])
            for pixel in sample_points]


if __name__ == '__main__':
    test_filename = 'bea24.jpg'
    auto_filename = join('../img/auto', test_filename)
    auto_image = io.imread(auto_filename)
    # print(auto_image)
    # lab_image = color.rgb2lab(rgb_image)
    color_points = get_color_points(test_filename)
    print(color_points)
    # print('Before modification')
    # print(lab_image)
    for pixel_ratio, lab in color_points:
        pixel = (int(pixel_ratio[0] * auto_image.shape[0]), int(pixel_ratio[1] * auto_image.shape[1]))
        auto_image[pixel] = [255, 0, 255]
    # print('After modification')
    # print(lab_image)
    io.imsave(join('../output', test_filename[:-3] + 'png'), auto_image)
