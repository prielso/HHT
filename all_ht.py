import numpy as np
from skimage import feature
from scipy import ndimage
from tqdm import tqdm
from skimage.filters import gaussian


def hht(data, num_of_hyperbolas, sigma, convx_min, convx_max, convx_resolution, min_a0=0, max_a0=None,
        a0a2_window_size=None):
    """
    This function transform the data edges to a parameters space of the Hyperbola geometric shape according to the
    mathematical expression- y = a0 + a1 * (x - a2)**2 .
    in this way of expressing an hyperbola, the parameters are:
    1. a0 - is the 'y' coordinate of the hyperbola peak (could be also referred as the peak's depth or row indice).
       the location of the peak we are looking for is inside the data (means that a0 is lower the data.shape[0]).
    2. a2 - is the 'x' coordinate of the hyperbola peak (could be also referred as the peak's coloumn indice).
       the location of the peak we are looking for is inside the data (means that a0 is lower the data.shape[1]).
    3. a1 - is the convexity of the hyperbola.
       the higher the value of a1 the sharper the hyperbola will look (more pointy).
       after ampirically observation of the a1 values we conclude that the range of a1 values needs to be defined by the
       user.
    :param data: numpy array.
    :param num_of_hyperbolas: how many hyperbolas to return (starting from the highest score hyperbola downwards).
    :param sigma: the first stage of the Canny algorithm (edges detector) is smoothing the data
    :param max_a0: the maximum possible value of the Hyperbola's peak y-coordinate.
    :param min_a0: the minimum possible value of the Hyperbola's peak y-coordinate.
    :param convx_resolution: the resulotion of the a1 values (e.g. 0.001 for 3 digits after the decimal point accuracy)
    :param convx_max: a1 maximum value to vote.
    :param convx_min: a1 minimum value to vote.
    :param a0a2_window_size: a list of 2 integers: [peak's row distance, peak's coloumn distance]
    first, the maximum number of rows possible between the edge pixel and the hyperbola's peak row.
    second, the maximum number of coulumns possible between the edge pixel and the hyperbola's peak row.
    :return: A list of the highest scored hyperbolas, from top to bottm (with #num_of_hyperbolas lentgh)
             Every element is [a0, a1, a2, score] for the specific hyperbola.
    """
    data_edges = feature.canny(data, sigma=sigma)
    if max_a0 is None:
        max_a0 = data.shape[0]
    if a0a2_window_size is None:
        a0_offset = data.shape[0]
        a2_offset = data.shape[1]
    else:
        a0_offset = a0a2_window_size[0]
        a2_offset = a0a2_window_size[1]
    a1_step = convx_resolution  # the accuracy of the a1 value.
    a1_max = convx_max  # highest a1 value we are looking for
    a1_min = convx_min  # lowest a1 value we are looking for
    a1_num = int((a1_max - a1_min) / a1_step) + 1  # number of different a1 values aka quantization of the a1 range.
    voting_space = np.zeros((data.shape[1], a1_num, data.shape[0]), dtype=np.int16)
    # A indices list of all True pixels in the edges image.
    edges_indices = np.where(data_edges == True)
    for i in tqdm(range(len(edges_indices[0]))):  # for every edge-pixel (True) in the data
        x = edges_indices[1][i]  # x-coordinate of the i's edged pixel
        y = edges_indices[0][i]  # y-coordinate of the i's edged pixel
        for a0 in range(max(0, y - a0_offset, min_a0), min(y, max_a0)):
            for a2 in range(max(0, x - a2_offset), min(x + 1 + a2_offset, data.shape[1])):
                if a2 == x:
                    continue
                a1 = (y - a0) / ((x - a2) ** 2)
                a1_error = 1 / ((x - a2) ** 2)
                if a1_min <= a1 + a1_error <= a1_max:
                    a1_idx = int((a1 - a1_min) / a1_step)
                    error_idx = int(a1_error / a1_step)
                    for j in range(error_idx):
                        if a1_idx + j in range(a1_num):
                            voting_space[a2, a1_idx + j, a0] += 1
    winning_parameters = []
    for j in range(num_of_hyperbolas):
        max_vote = np.amax(voting_space)
        max_voting_idx = np.where(voting_space == np.amax(voting_space))
        a0 = max_voting_idx[2][0]
        a1 = a1_min + convx_resolution * max_voting_idx[1][0]
        a2 = max_voting_idx[0][0]
        voting_space[max_voting_idx[0][0], max_voting_idx[1][0], max_voting_idx[2][0]] = 0
        winning_parameters.append([a0, a1, a2, max_vote])
    return winning_parameters


### other versions of the HHT that differ from each other in the order of the hyperbola's parameters:
def ht_hyperbola_slow_a0_a1(data, num_of_hyperbolas, sigma, convx_min, convx_max, convx_resolution, min_a0=0,
                            max_a0=None):
    data_edges = feature.canny(data, sigma=sigma)
    a1_step = convx_resolution  # the accuracy of the a1 value.
    a1_max = convx_max  # highest a1 value we are looking for
    a1_min = convx_min  # lowest a1 value we are looking for
    a1_num = int((a1_max - a1_min) / a1_step) + 1  # number of different a1 values aka quantization of the a1 range.
    if max_a0 is None:
        max_a0 = data.shape[0]
    voting_space = np.zeros((data.shape[1], a1_num, data.shape[0]), dtype=np.int16)
    num_neighbors = 1
    # A indices list of all True pixels in the edges image.
    edges_indices = np.where(data_edges == True)
    for i in tqdm(range(len(edges_indices[0]))):  # for every edge-pixel (True) in the data
        x = edges_indices[1][i]  # x-coordinate of the i's edged pixel
        y = edges_indices[0][i]  # y-coordinate of the i's edged pixel
        for a0 in range(min_a0, min(y + 1, max_a0)):  # for every possible value of a0 parameter (peak's row)
            for a1_idx in range(a1_num):  # for every possible value of a1 parameter (hyperbola convexity)
                a1 = a1_idx * a1_step + a1_min
                horizontal_offset = np.sqrt((y - a0) / a1)
                a2_1 = x - horizontal_offset
                a2_2 = x + horizontal_offset
                if int(a2_1) + 1 in range(data.shape[1]):
                    voting_space[int(a2_1) + 1, a1_idx, a0] += 1
                if int(a2_1) in range(data.shape[1]):
                    voting_space[int(a2_1), a1_idx, a0] += 1
                if int(a2_2) + 1 in range(data.shape[1]):
                    voting_space[int(a2_2) + 1, a1_idx, a0] += 1
                if int(a2_2) in range(data.shape[1]):
                    voting_space[int(a2_2), a1_idx, a0] += 1

    winning_parameters = []
    for j in range(num_of_hyperbolas):
        max_vote = np.amax(voting_space)
        max_voting_idx = np.where(voting_space == np.amax(voting_space))
        a0 = max_voting_idx[2][0]
        a1 = a1_min + convx_resolution * max_voting_idx[1][0]
        a2 = max_voting_idx[0][0]
        voting_space[max_voting_idx[0][0], max_voting_idx[1][0], max_voting_idx[2][0]] = 0
        winning_parameters.append([a0, a1, a2, max_vote])
    return winning_parameters


def ht_hyperbola_slow_a0_a2(data, num_of_hyperbolas, sigma, convx_min, convx_max, convx_resolution, min_a0=0,
                            max_a0=None):
    data_edges = feature.canny(data, sigma=sigma)
    a1_step = convx_resolution  # the accuracy of the a1 value.
    a1_max = convx_max  # highest a1 value we are looking for
    a1_min = convx_min  # lowest a1 value we are looking for
    a1_num = int((a1_max - a1_min) / a1_step) + 1  # number of different a1 values aka quantization of the a1 range.
    if max_a0 is None:
        max_a0 = data.shape[0]
    voting_space = np.zeros((data.shape[1], a1_num, data.shape[0]), dtype=np.int16)

    # A indices list of all True pixels in the edges image.
    edges_indices = np.where(data_edges == True)
    for i in tqdm(range(len(edges_indices[0]))):  # for every edge-pixel (True) in the data
        x = edges_indices[1][i]  # x-coordinate of the i's edged pixel
        y = edges_indices[0][i]  # y-coordinate of the i's edged pixel
        for a0 in range(min_a0, min(y, max_a0)):  # for every possible value of a0 parameter (peak's row)
            for a2 in range(data.shape[1]):  # for every possible value of a1 parameter (hyperbola convexity)
                if a2 == x:
                    continue
                a1 = (y - a0) / ((x - a2) ** 2)
                a1_error = 1 / ((x - a2) ** 2)
                error_idx = int(a1_error / a1_step)
                if a1_min <= a1 <= a1_max:
                    a1_idx = int((a1 - a1_min) / a1_step)
                    for j in range(error_idx):
                        if a1_idx + j in range(a1_num):
                            voting_space[a2, a1_idx + j, a0] += 1
    winning_parameters = []
    for j in range(num_of_hyperbolas):
        max_vote = np.amax(voting_space)
        max_voting_idx = np.where(voting_space == np.amax(voting_space))
        a0 = max_voting_idx[2][0]
        a1 = a1_min + convx_resolution * max_voting_idx[1][0]
        a2 = max_voting_idx[0][0]
        voting_space[max_voting_idx[0][0], max_voting_idx[1][0], max_voting_idx[2][0]] = 0
        winning_parameters.append([a0, a1, a2, max_vote])
    return winning_parameters


def ht_hyperbola_faster_a0_a2(data, num_of_hyperbolas, sigma, convx_min, convx_max, convx_resolution, min_a0=0,
                              max_a0=None):
    data_edges = feature.canny(data, sigma=sigma)
    data_blurred = gaussian(data, sigma=sigma)
    sobel_y = ndimage.sobel(data_blurred, axis=0, mode='constant')
    sobel_x = ndimage.sobel(data_blurred, axis=1, mode='constant')
    eps = 1e-9
    # m = sobel_y / (sobel_x + eps)  # equivalent to y' (dy/dx)
    theta = np.arctan2(sobel_y, sobel_x)
    m = np.zeros(data.shape)
    m[np.where(data_edges == True)] = np.tan(theta - np.pi / 2)[np.where(data_edges == True)]

    a1_step = convx_resolution  # the accuracy of the a1 value.
    a1_max = convx_max  # highest a1 value we are looking for
    a1_min = convx_min  # lowest a1 value we are looking for
    a1_num = int((a1_max - a1_min) / a1_step) + 1  # number of different a1 values aka quantization of the a1 range.
    if max_a0 is None:
        max_a0 = data.shape[0]
    voting_space = np.zeros((data.shape[1], a1_num, data.shape[0]), dtype=np.int16)
    num_neighbors = 1
    # A indices list of all True pixels in the edges image.
    edges_indices = np.where(data_edges == True)
    for i in tqdm(range(len(edges_indices[0]))):  # for every edge-pixel (True) in the data
        x = edges_indices[1][i]  # x-coordinate of the i's edged pixel
        y = edges_indices[0][i]  # y-coordinate of the i's edged pixel
        if m[y, x] == 0:
            continue
        for a0 in range(min_a0, min(y, max_a0)):  # for every possible value of a0 parameter (peak's row)
            if 0 < m[y, x]:
                for a2 in range(x):  # for every possible value of a1 parameter (hyperbola convexity)
                    a1 = (y - a0) / ((x - a2) ** 2)
                    a1_error = 1 / ((x - a2) ** 2)
                    if a1_min <= a1 + a1_error <= a1_max:
                        a1_idx = int((a1 - a1_min) / a1_step)
                        error_idx = int(a1_error / a1_step)
                        for j in range(error_idx):
                            if a1_idx + j in range(a1_num):
                                voting_space[a2, a1_idx + j, a0] += 1
            elif m[y, x] < 0:
                for a2 in range(x + 1, data.shape[1]):
                    a1 = (y - a0) / ((x - a2) ** 2)
                    a1_error = 1 / ((x - a2) ** 2)
                    if a1_min <= a1 + a1_error <= a1_max:
                        a1_idx = int((a1 - a1_min) / a1_step)
                        error_idx = int(a1_error / a1_step)
                        for j in range(error_idx):
                            if a1_idx + j in range(a1_num):
                                voting_space[a2, a1_idx + j, a0] += 1
    winning_parameters = []
    for j in range(num_of_hyperbolas):
        max_vote = np.amax(voting_space)
        max_voting_idx = np.where(voting_space == np.amax(voting_space))
        a0 = max_voting_idx[2][0]
        a1 = a1_min + convx_resolution * max_voting_idx[1][0]
        a2 = max_voting_idx[0][0]
        voting_space[max_voting_idx[0][0], max_voting_idx[1][0], max_voting_idx[2][0]] = 0
        winning_parameters.append([a0, a1, a2, max_vote])
    return winning_parameters


def ht_hyperbola_a0_a2_local(data, num_of_hyperbolas, sigma, convx_min, convx_max, convx_resolution, min_a0=0,
                             max_a0=None, a0a2_window_size=None):
    data_edges = feature.canny(data, sigma=sigma)
    data_blurred = gaussian(data, sigma=sigma)
    sobel_y = ndimage.sobel(data_blurred, axis=0, mode='constant')
    sobel_x = ndimage.sobel(data_blurred, axis=1, mode='constant')
    eps = 1e-9
    # m = sobel_y / (sobel_x + eps)  # equivalent to y' (dy/dx)
    theta = np.arctan2(sobel_y, sobel_x)
    m = np.zeros(data.shape)
    m[np.where(data_edges == True)] = np.tan(theta - np.pi / 2)[np.where(data_edges == True)]
    if a0a2_window_size is None:
        a0_offset = data.shape[0]
        a2_offset = data.shape[1]
    else:
        a0_offset = a0a2_window_size[0]
        a2_offset = a0a2_window_size[1]
    if max_a0 is None:
        max_a0 = data.shape[0]
    a1_step = convx_resolution  # the accuracy of the a1 value.
    a1_max = convx_max  # highest a1 value we are looking for
    a1_min = convx_min  # lowest a1 value we are looking for
    a1_num = int((a1_max - a1_min) / a1_step) + 1  # number of different a1 values aka quantization of the a1 range.
    voting_space = np.zeros((data.shape[1], a1_num, data.shape[0]), dtype=np.int16)
    num_neighbors = 1
    # A indices list of all True pixels in the edges image.
    edges_indices = np.where(data_edges == True)
    for i in tqdm(range(len(edges_indices[0]))):  # for every edge-pixel (True) in the data
        x = edges_indices[1][i]  # x-coordinate of the i's edged pixel
        y = edges_indices[0][i]  # y-coordinate of the i's edged pixel
        if m[y, x] == 0:
            continue
        for a0 in range(max(0, y - a0_offset, min_a0), min(y, max_a0)):
            if 0 < m[y, x]:
                for a2 in range(max(0, x - a2_offset), x):  # for every possible value of a1 parameter (convexity)
                    a1 = (y - a0) / ((x - a2) ** 2)
                    a1_error = 1 / ((x - a2) ** 2)
                    if a1_min <= a1 + a1_error <= a1_max:
                        a1_idx = int((a1 - a1_min) / a1_step)
                        error_idx = int(a1_error / a1_step)
                        for j in range(error_idx):
                            if a1_idx + j in range(a1_num):
                                voting_space[a2, a1_idx + j, a0] += 1
            elif m[y, x] < 0:
                for a2 in range(x + 1, min(x + 1 + a2_offset, data.shape[1])):
                    a1 = (y - a0) / ((x - a2) ** 2)
                    a1_error = 1 / ((x - a2) ** 2)
                    if a1_min <= a1 + a1_error <= a1_max:
                        a1_idx = int((a1 - a1_min) / a1_step)
                        error_idx = int(a1_error / a1_step)
                        for j in range(error_idx):
                            if a1_idx + j in range(a1_num):
                                voting_space[a2, a1_idx + j, a0] += 1
    winning_parameters = []
    for j in range(num_of_hyperbolas):
        max_vote = np.amax(voting_space)
        max_voting_idx = np.where(voting_space == np.amax(voting_space))
        a0 = max_voting_idx[2][0]
        a1 = a1_min + convx_resolution * max_voting_idx[1][0]
        a2 = max_voting_idx[0][0]
        voting_space[max_voting_idx[0][0], max_voting_idx[1][0], max_voting_idx[2][0]] = 0
        winning_parameters.append([a0, a1, a2, max_vote])
    return winning_parameters


def ht_hyperbola_fast(data, num_of_hyperbolas, sigma, convx_min, convx_max, convx_resolution, min_a0=0, max_a0=None):
    data_blurred = gaussian(data, sigma=sigma)
    sobel_y = ndimage.sobel(data_blurred, axis=0, mode='constant')
    sobel_x = ndimage.sobel(data_blurred, axis=1, mode='constant')
    eps = 1e-9
    # m = sobel_y / (sobel_x + eps)  # equivalent to y' (dy/dx)
    theta = np.arctan2(sobel_y, sobel_x)
    m = np.tan(theta - np.pi / 2)
    data_edges = feature.canny(data, sigma=sigma)  # extracting edges using canny
    a1_step = convx_resolution  # the accuracy of the a1 value.
    a1_max = convx_max  # highest a1 value we are looking for
    a1_min = convx_min  # lowest a1 value we are looking for
    a1_num = int((a1_max - a1_min) / a1_step) + 1  # number of different a1 values aka quantization of the a1 range.
    voting_space = np.zeros((data_edges.shape[1], a1_num, data_edges.shape[0]), dtype=np.int16)  # [a2, a1 ,a0]
    vote_area = [1, 5, 3]
    edges_indices = np.where(data_edges == True)  # A indices list of all True pixels in the edges image.
    for i in tqdm(range(len(edges_indices[0]))):  # for every edge-pixel (True) in the data
        x = edges_indices[1][i]
        y = edges_indices[0][i]
        if m[y, x] == 0:
            continue
        for a0 in range(min_a0, min(y + 1, max_a0)):  # for every possible value of a0 parameter (peak's row)
            if a0 == y:
                voting_space[x, :, y] += 1
                continue
            a1 = (m[y, x] ** 2) / (4 * (y - a0))
            a2 = x - 2 * (y - a0) / m[y, x]
            a1_idx = int(a1 / a1_step) - 1
            a2_idx = int(a2)
            if (0 <= a1_idx < a1_num) and (0 <= a2_idx < data.shape[1]):
                for i0 in range(vote_area[0]):
                    for i1 in range(vote_area[1]):
                        for i2 in range(vote_area[2]):
                            a2_n_idx = (a2_idx - int(vote_area[0] / 2) + i0)
                            a1_n_idx = (a1_idx - int(vote_area[1] / 2) + i1)
                            a0_n_idx = (a0 - int(vote_area[2] / 2) + i2)
                            if ((0 <= a2_n_idx < voting_space.shape[0]) and
                                    (0 <= a1_n_idx < voting_space.shape[1]) and
                                    (0 <= a0_n_idx < voting_space.shape[2])):
                                voting_space[a2_n_idx, a1_n_idx, a0_n_idx] += 1
    winning_parameters = []
    for j in range(num_of_hyperbolas):
        max_vote = np.amax(voting_space)
        max_voting_idx = np.where(voting_space == np.amax(voting_space))
        a0 = max_voting_idx[2][0]
        a1 = a1_min + convx_resolution * max_voting_idx[1][0]
        a2 = max_voting_idx[0][0]
        voting_space[max_voting_idx[0][0], max_voting_idx[1][0], max_voting_idx[2][0]] = 0
        winning_parameters.append([a0, a1, a2, max_vote])
    return winning_parameters
