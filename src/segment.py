import scipy.misc
import scipy.signal
import numpy as np
import random
from collections import deque
from PIL import Image
from skimage import io, color
from os.path import join
import cv2
from sklearn.cluster import KMeans

class Region(object):
    def __init__(self, lab_image, unlabeled_pixels_set, id):
        self.id = id
        self.lab_image = lab_image
        self.shape = lab_image.shape
        seed = random.sample(unlabeled_pixels_set, 1)[0]
        self.pixels = set()
        self.connected_regions = {}
        self.color = None

        self.verbose = False
        if self.verbose:
            print('---------------------Begin segmenting one new region')
            print('Seed = %s' % (seed,))
        newly_labeled = deque()
        newly_labeled.append(seed)
        self.pixels.add(seed)
        unlabeled_pixels_set.remove(seed)
        while len(newly_labeled) > 0:
            pixel = newly_labeled.popleft()
            if self.verbose:
                print('Got pixel %s from queue' % (pixel,))
            for adjacent_pixel in self.adjacent_unlabeled_pixels(pixel, unlabeled_pixels_set):
                self.pixels.add(adjacent_pixel)
                unlabeled_pixels_set.remove(adjacent_pixel)
                newly_labeled.append(adjacent_pixel)
                # self.check_rep()

    def connect_to(self, region, gradient):
        assert region != self
        if region not in self.connected_regions:
            self.connected_regions[region] = (1, gradient)
        else:
            new_num = self.connected_regions[region][0]
            new_weight = (self.connected_regions[region][1] + gradient) / (new_num * 1.0)
            self.connected_regions[region] = (new_num, new_weight)

    def get_color(self):
        return self.color

    def decide_color(self):
        assert self.color is None
        bad_colors = {}
        for region, weight in self.connected_regions:
            if region.get_color() is not None and weight > 10:
                bad_colors[region.get_color()] = weight ** 0.5

        abs = np.zeros((len(self.pixels), 2))
        abs = []
        for pixel in self.pixels:
            ab = self.lab_image[pixel][1:3]
            close_to_adjacent_colors = False
            for bad_color, radius in bad_colors:
                if ((ab[0] - bad_color[0])**2 + (ab[1] - bad_color[1])**2)**0.5 < radius:
                    close_to_adjacent_colors = True
            if not close_to_adjacent_colors:
                abs.append(ab)

        if len(abs) == 0:
            print('Region %d with size %d gets zero pixels remained' % (self.id, len(self.pixels)))
            # TODO: may change color here
            pixel = random.sample(self.pixels, 1)[0]
            abs.append(self.lab_image[pixel][1:3])

        abs = np.array(abs)
        kmeans = KMeans(n_clusters=8).fit(abs)
        numbers = [0] * kmeans.cluster_centers_.shape[0]
        for label in kmeans.labels_:
            numbers[label] += 1
        chosen_color = kmeans.cluster_centers_[numbers.index(max(numbers))]
        return tuple(chosen_color)

    def adjacent_unlabeled_pixels(self, pixel, unlabeled_pixels_set, mode='cross'):
        assert type(pixel) is tuple
        row = pixel[0]
        col = pixel[1]
        if mode == 'cross':
            pixels = [(row, col), (row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]
            return [p for p in pixels if p in unlabeled_pixels_set]
        elif mode == '3x3':
            rows = range(pixel[0] - 1, pixel[0] + 2)
            cols = range(pixel[1] - 1, pixel[1] + 2)
            return [(row, col) for row in rows for col in cols if (row, col) in unlabeled_pixels_set]

    def __hash__(self):
        return id


class ImageSegment(object):
    def __init__(self, bw_image_filename, color_image_filename, verbose=False):
        self.bw_image = scipy.misc.imread(bw_image_filename, mode='L')
        self.color_image = scipy.misc.imread(color_image_filename, mode='RGB')
        self.lab_image = color.rgb2lab(self.color_image)
        assert self.bw_image.shape == self.color_image.shape
        self.shape = self.bw_image.shape
        self.verbose = verbose
        self.unlabeled = set((row, col) for row in range(self.shape[0]) for col in range(self.shape[1]))
        self.gradient = self.calculate_gradient()
        self.threshold = 100
        self.is_edge = cv2.Canny(self.bw_image, 50, 200)
        self.label = -1 - self.is_edge # All edges are labeled -256, unlabeled are -1
        self.regions = []

    def check_rep(self):
        for label in range(len(self.segmentation)):
            for pixel in self.segmentation[label]:
                if label == 1:
                    print('checking pixel %s of label %d' % (pixel, label))
                assert type(pixel) is tuple
                assert len(pixel) == 2

    def calculate_gradient(self):
        """Calculate the gradient matrix"""
        kernel_x = np.array([[+1, 0, -1],
                             [+2, 0, -2],
                             [+1, 0, -1]])
        kernel_y = np.array([[+1, +2, +1],
                             [ 0,  0,  0],
                             [-1, -2, -1]])
        gradient_x = scipy.signal.convolve2d(self.bw_image, kernel_x, mode='same', boundary='symm')
        gradient_y = scipy.signal.convolve2d(self.bw_image, kernel_y, mode='same', boundary='symm')
        return np.sqrt(gradient_x ** 2 + gradient_y ** 2)

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

    def segment(self):
        if self.verbose:
            print('Begin segmenting')
        while len(self.unlabeled) != 0:
            if self.verbose:
                print('There are %d unlabeled pixels left' % len(self.unlabeled))
            new_region = Region(self.lab_image, self.unlabeled, len(self.regions))
            self.regions.append(new_region)
        if self.verbose:
            print('Finished segmenting')

    def choose_color(self):
        self.regions.sort(self.regions, key=lambda region: region.)

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
    image_segment.save_segmentation(join('../img/segment', filename[:-3] + 'png'))

    rgb = io.imread(auto_filename)
    lab = color.rgb2lab(rgb)
    return [((1.0 * pixel[0]/image_segment.shape[0], 1.0 * pixel[1]/image_segment.shape[1]), lab[pixel])
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
