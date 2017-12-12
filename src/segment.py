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

dict = {(-1, -1): [(-2, -2), (-2, -1), (-2, 0), (-1, -2), (0, -2)],
        (1, 1): [(2, 2), (2, 1), (2, 0), (1, 2), (0, 2)],
        (-1, 1): [(-2, 2), (-2, 1), (-2, 0), (-1, 2), (0, 2)],
        (1, -1): [(2, -2), (2, -1), (2, 0), (1, -2), (0, -2)],
        (1, 0): [(2, 0), (2, -1), (2, 1)],
        (-1, 0): [(-2, 0), (-2, -1), (-2, 1)],
        (0, 1): [(0, 2), (-1, 2), (1, 2)],
        (0, -1): [(0, -2), (-1, -2), (1, -2)]}

class Region(object):
    def __init__(self, lab_image, unlabeled_pixels_set, ID):
        assert type(ID) == int
        self.ID = ID
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
        self.size = len(self.pixels)

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
        # TODO: decide bad colors
        # for region, link in self.connected_regions.items():
        #     if region.get_color() is not None and link[1] > 10:
        #         bad_colors[region.get_color()] = link[1] ** 0.5

        abs = []
        for pixel in self.pixels:
            ab = self.lab_image[pixel][1:3]
            close_to_adjacent_colors = False
            for bad_color, radius in bad_colors.items():
                if ((ab[0] - bad_color[0])**2 + (ab[1] - bad_color[1])**2)**0.5 < radius:
                    close_to_adjacent_colors = True
            if not close_to_adjacent_colors:
                abs.append(ab)

        if len(abs) == 0:
            print('Region %d with size %d gets zero pixels remained' % (self.ID, len(self.pixels)))
            # TODO: may change color here
            pixel = random.sample(self.pixels, 1)[0]
            abs.append(self.lab_image[pixel][1:3])

        abs = np.array(abs)
        n_clusters = min(abs.shape[0], 8)
        kmeans = KMeans(n_clusters=n_clusters).fit(abs)
        numbers = [0] * kmeans.cluster_centers_.shape[0]
        for label in kmeans.labels_:
            numbers[label] += 1
        chosen_color = kmeans.cluster_centers_[numbers.index(max(numbers))]
        self.color = tuple(chosen_color)
        assert type(self.color) == tuple
        assert len(self.color) == 2

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
        assert type(self.ID) == int
        return self.ID

    def get_sample_points(self):
        assert self.color is not None
        if self.size < 10:
            return []
        num = max(1, int(self.size / 1000))
        print('sample %d points for region %d' % (num, self.ID))
        points = random.sample(self.pixels, num)
        return [((point[0] * 1.0 / self.shape[0], point[1] * 1.0 / self.shape[1]), (0, self.color[0], self.color[1])) for point in points]


class ImageSegment(object):
    def __init__(self, bw_image_filename, color_image_filename, verbose=False):
        self.bw_image = scipy.misc.imread(bw_image_filename, mode='L')
        self.color_image = scipy.misc.imread(color_image_filename, mode='RGB')
        self.lab_image = color.rgb2lab(self.color_image)
        assert self.bw_image.shape == self.color_image.shape[0:2]
        self.shape = self.bw_image.shape
        self.verbose = verbose
        self.gradient = self.calculate_gradient()
        self.threshold = 100
        self.is_edge = cv2.Canny(self.bw_image, 50, 200)
        self.label = -1 - self.is_edge # All edges are labeled -256, unlabeled are -1
        self.regions = []
        self.fix_edge()
        self.unlabeled = set((row, col) for row in range(self.shape[0]) for col in range(self.shape[1])
                             if not self.is_edge[row, col])

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

    def segment(self):
        if self.verbose:
            print('Begin segmenting')
        while len(self.unlabeled) != 0:
            if self.verbose:
                print('There are %d unlabeled pixels left' % len(self.unlabeled))
            ID = len(self.regions)
            new_region = Region(self.lab_image, self.unlabeled, ID)
            self.regions.append(new_region)
            for pixel in new_region.pixels:
                self.label[pixel] = ID
        if self.verbose:
            print('Finished segmenting')

    def connect(self):
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                if self.is_edge[row, col]:
                    connected_id = set()
                    for i in [-1, 0, 1]:
                        for j in [-1, 0, 1]:
                            if 0 <= row + i < self.shape[0] and 0 <= col + j < self.shape[1]:
                                if self.label[(row + i, col + j)] >= 0:
                                    connected_id.add(self.label[(row + i, col + j)])
                    if len(connected_id) == 2:
                        id_list = list(connected_id)
                        region_1 = self.regions[id_list[0]]
                        region_2 = self.regions[id_list[1]]
                        gradient = self.gradient[row, col]
                        region_1.connect_to(region_2, gradient)
                        region_2.connect_to(region_1, gradient)

    def choose_color(self):
        self.regions.sort(key=lambda region: region.size)
        for region in self.regions:
            region.decide_color()

    def random_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def save_segmentation(self, output_filename):
        for region in self.regions:
            ab = region.get_color()
            for pixel in region.pixels:
                self.lab_image[pixel][1:3] = ab

        rgb = color.lab2rgb(self.lab_image) * 255
        rgb = rgb.astype(np.uint8)
        # io.concatenate_images((self.color_image, self.bw_image, self.gradient, rgb))
        auto_image = Image.fromarray(self.color_image)
        bw_image = Image.fromarray(self.bw_image)
        gradient_image = Image.fromarray(self.gradient)
        edge_image = Image.fromarray(self.is_edge)
        segment_image = Image.fromarray(rgb)

        random_image = Image.new("RGB", (self.shape[1], self.shape[0]))
        pixels = random_image.load()
        for region in self.regions:
            random_color = self.random_color()
            for pixel in region.pixels:
                pixels[pixel[1], pixel[0]] = random_color

        comparison_image = Image.new("RGB", (self.shape[1] * 6, self.shape[0]))
        comparison_image.paste(auto_image, (0, 0))
        comparison_image.paste(bw_image, (self.shape[1], 0))
        comparison_image.paste(gradient_image, (self.shape[1] * 2, 0))
        comparison_image.paste(edge_image, (self.shape[1] * 3, 0))
        comparison_image.paste(segment_image, (self.shape[1] * 4, 0))
        comparison_image.paste(random_image, (self.shape[1] * 5, 0))
        comparison_image.save(output_filename)

    def get_sample_points(self):
        sample_points = []
        # Sample one point from all labeled sets except label 0 (unlabeled) and label 1 (edge)
        for region in self.regions:
            sample_points += region.get_sample_points()
        return sample_points

    def run(self):
        self.segment()
        self.connect()
        self.choose_color()
        return self.get_sample_points()

        # self.calculate_gradient()
        # self.segment()
        # return self.get_sample_points()
        # # self.show_segment()


    def fix_edge(self):
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                direction = self.potential_direction(row, col)
                if direction is None:
                    continue
                if self.verbose:
                    print('direction is not None for pixel: %s' % ((row, col),))
                opposite_direction = self.opposite_direction(direction)
                opposite_pixel = self.add_pixels((row, col), opposite_direction)
                if not (0 <= opposite_pixel[0] < self.shape[0] and 0 <= opposite_pixel[1] < self.shape[1]):
                    continue
                for d in self.positions(opposite_direction):
                    check_pixel = self.add_pixels((row, col), d)
                    if 0 <= check_pixel[0] < self.shape[0] and 0 <= check_pixel[1] < self.shape[1]:
                        if self.verbose:
                            print('Got one check_pixel %s for pixel: %s' % (check_pixel, (row, col),))
                        if self.is_edge[check_pixel]:
                            if self.verbose:
                                print('Add one edge at %s for pixel: %s' % (self.add_pixels((row, col), direction), (row, col),))
                            self.is_edge[self.add_pixels((row, col), opposite_direction)] = 255

    def positions(self, direction):
        return dict[direction]

    def potential_direction(self, row, col):
        if self.is_edge[row, col]:
            num = 0
            direction = None
            for d in [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1), (1, 0), (1, 1)]:
                pixel = self.add_pixels((row, col), d)
                if 0 <= pixel[0] < self.shape[0] and 0 <= pixel[1] < self.shape[1]:
                    if self.is_edge[pixel]:
                        num += 1
                        direction = d
            if num == 1:
                return direction
        return None

    def opposite_direction(self, direction):
        return (-direction[0], -direction[1])

    def add_pixels(self, t1, t2):
        return (t1[0] + t2[0], t1[1] + t2[1])


def get_color_points(filename):
    auto_filename = join('../img/auto', filename)
    bw_filename = join('../img/bw', filename)
    image_segment = ImageSegment(bw_filename, auto_filename, verbose=False)
    sample_points = image_segment.run()
    image_segment.save_segmentation(join('../img/segment', filename[:-3] + 'png'))

    return sample_points


if __name__ == '__main__':
    test_filename = 'better/dog4.jpg'
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
