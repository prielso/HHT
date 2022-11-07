import numpy as np
import matplotlib.pyplot as plt
from all_ht import *
import cv2
import time
from PIL import Image


def main():
    data = np.load('example_2d_array.npy')
    sigma = (3, 3)
    canny = feature.canny(data, sigma=sigma)

    # show the data and its edges:
    f, images = plt.subplots(1, 2)
    plt.gray()
    images[0].imshow(data, aspect='auto')
    images[0].set_title('Numpy Array Data')
    images[1].imshow(canny, aspect='auto')
    images[1].set_title('Numpy Array Data Edges\n(Using Canny Algorithm):')
    plt.show()

    # convert the data to image to present the results:
    data_tested_int32 = np.int32(data)
    normalized_data = (data_tested_int32 - np.amin(data_tested_int32)) / (
            np.amax(data_tested_int32) - np.amin(data_tested_int32))
    image_color = cv2.cvtColor(np.uint8(normalized_data * 255), cv2.COLOR_GRAY2RGB)
    image_data = Image.fromarray(image_color)

    a1_min = 0.01
    a1_max = 0.1
    a1_res = 0.01
    min_a0 = 100
    max_a0 = None
    a0a2_window = [100, 70]
    start_time = time.time()

    ################### Running The Algorithm!! ##############################
    winning_parameters_list = hht(data=data, num_of_hyperbolas=10, sigma=sigma,
                                  convx_min=a1_min, convx_max=a1_max,
                                  convx_resolution=a1_res, min_a0=min_a0,
                                  max_a0=max_a0, a0a2_window_size=a0a2_window)
    ###########################################################################
    execution_time = time.time() - start_time

    for hyperbola_i in range(len(winning_parameters_list)):
        a0_result = winning_parameters_list[hyperbola_i][0]
        a1_result = winning_parameters_list[hyperbola_i][1]
        a2_result = winning_parameters_list[hyperbola_i][2]
        the_score = winning_parameters_list[hyperbola_i][3]
        print(a0_result, a1_result, a2_result)
        draw_result, draw_result_dy = hyperbola_draw(a0_result, a1_result, a2_result, data.shape)
        combined_result = np.zeros(data.shape)
        combined_result[np.where(canny == True)] = 100
        combined_result[np.where(draw_result == 100)] += 155

        for y in range(draw_result.shape[0]):
            for x in range(draw_result.shape[1]):
                if draw_result[y, x] == 100 and canny[y, x] == True and (
                        abs(x - a2_result) < 2 * a0a2_window[1]):
                    image_data.putpixel((x, y), (255, 0, 0))

        f, images = plt.subplots(1, 2)
        plt.gray()
        images[1].imshow(combined_result, aspect='auto')
        images[1].set_title('Canny Edges With The Detected Hyperbola:')
        images[0].set_title('Hyperbola ' + str(hyperbola_i + 1) + '/' + str(
            len(winning_parameters_list)) + '\n' +
                            'a0= ' + str(a0_result) + ', ' +
                            'a1= ' + str(np.float16(a1_result)) + ', ' +
                            'a2= ' + str(a2_result) + ',\n score: ' + str(the_score) +
                            ', Run time: ' + str(np.float16(execution_time / 60)) + '[min]')
        images[0].imshow(image_data, aspect='auto')
        plt.show()


def hyperbola_draw(a0, a1, a2, dim):
    drawing = np.zeros(dim)
    drawing_dy = np.zeros(dim)
    for x in range(dim[1]):
        y = a0 + a1 * (x - a2) ** 2
        dy = 2 * a1 * (x - a2)
        if int(y) < dim[0]:
            drawing[int(y), x] = 100
            drawing_dy[int(y), x] = dy
    return drawing, drawing_dy


if __name__ == '__main__':
    main()
