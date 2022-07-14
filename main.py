import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
import cv2
import re
from skimage import feature
from scipy import ndimage
from all_ht import *
from Bscan_utils import *


def main():
    ## The real GPR data file location and name.
    path = "D://GPR_local/new_recordings/EB220613/"
    file_name = 't1s3r1'

    ## The synthetic GPR data file location and name.
    sim_path = "C://Users/prielso/gprMax/user_models/half_cylinder/"
    sim_file_name = 'half_cylinder_merged.out'

    sim_data, dt = get_output_data(sim_path+sim_file_name, 1, 'Ey')

    # Open output file and read number of outputs (receivers)
    # reading the meta data of the recording like: number of vertical samples, horizontal distance, time-window of the
    # signal. meaning for how long we sampled the returning signal.
    meta_data = open(path + file_name+'.rad', 'r+')
    meta_data = meta_data.readlines()
    samples = ''.join(meta_data[0])
    num_samples = int(re.findall("\d+", samples)[0])
    time_window = ''.join(meta_data[18])
    num_time_window = float(re.findall("\d+", time_window)[0])
    distance = ''.join(meta_data[23])
    num_distance = float(re.findall("\d+", distance)[0])

    data = open(path + file_name+'.rd3', 'rb')
    data = np.fromfile(data, dtype=np.int16)
    data = data.reshape(num_samples, int(data.shape[0] / num_samples), order='F')
    orig_data = data

    # gain correction, see decay_correction for more details
    gained_data = decay_correction(data, -120, 1, 30)
    gained_sim = decay_correction(sim_data, 7000, 10, 0)

    # CANNY edges in the image, using smoothing and then running Canny.
    real_sigma = [1, 0.25]
    sim_sigma = [4, 4]
    real_canny = feature.canny(gained_data, sigma=real_sigma)
    sim_canny = feature.canny(gained_sim, sigma=sim_sigma)

    # # drawing a sketch of an hyperbola for testing the algorithm
    # drawing_dim = [200, 100]
    # a0 = int(drawing_dim[0]/2)
    # a1 = 0.0224
    # a2 = int(drawing_dim[1]/2)
    # drawing, drawing_dy = hyperbola_draw(a0, a1, a2, drawing_dim)
    # drawing_blurred = gaussian(drawing, sigma=sim_sigma)
    # drawing_canny = feature.canny(drawing_blurred, sigma=sim_sigma)
    # # ###########################################################

    # sobel_y = ndimage.sobel(drawing_blurred, axis=0, mode='constant')
    # sobel_x = ndimage.sobel(drawing_blurred, axis=1, mode='constant')
    # eps = 1e-9
    # m_test = sobel_y / (sobel_x + eps)  # equivalent to y' (dy/dx)
    # theta = np.arctan(drawing_dy)
    # m_test[np.where(drawing != 255)] = 0
    # m_test_theta = np.arctan(m_test)
    # data_blurred = gaussian(data, sigma=real_sigma)

    plt.imshow(gained_data, cmap='gray')
    plt.title('Recording: ' + file_name)
    plt.show()

    plt.imshow(real_canny, cmap='gray')  # , vmin=0, vmax=255)
    plt.title(r'$\sigma$x: ' + str(real_sigma[0]) + r' $\sigma$y: ' + str(real_sigma[1]))
    plt.show()

    hyp_check, hyp_check_dy = hyperbola_draw(28, 0.001, 170, data.shape)
    hyp_check[np.where(real_canny == True)] += 155
    score = np.count_nonzero(hyp_check == 255)

    plt.imshow(hyp_check, cmap='gray')  # , vmin=0, vmax=255)
    plt.title('score is: ' + str(score))
    plt.show()

    # which data u r testing?
    data_tested = gained_data
    data_tested_sigma = real_sigma
    data_tested_canny = real_canny

    data_tested_int32 = np.int32(data_tested)
    normalized_data = (data_tested_int32 - np.amin(data_tested_int32)) / (
                        np.amax(data_tested_int32) - np.amin(data_tested_int32))
    image_color = cv2.cvtColor(np.uint8(normalized_data * 255), cv2.COLOR_GRAY2RGB)
    image_color_arr = Image.fromarray(image_color)

    convexity_min = 1e-3
    convexity_max = 2e-3
    convexity_res = 1e-5
    result, result_score = ht_hyperbola_faster_a0_a2(data=data_tested, num_of_hyperbolas=1, sigma=data_tested_sigma,
                                                   convx_min=convexity_min, convx_max=convexity_max, convx_resolution=convexity_res)
    print(result)
    print('score is: ' + str(result_score))

    for hyperbola_i in range(len(result[0])):
        a0_result = result[2][hyperbola_i]
        a1_result = convexity_min + convexity_res*result[1][hyperbola_i]
        a2_result = result[0][hyperbola_i]
        print(a0_result, a1_result, a2_result)
        draw_result, draw_result_dy = hyperbola_draw(a0_result, a1_result, a2_result, data_tested.shape)
        combined_result = np.zeros(data_tested.shape)
        combined_result[np.where(data_tested_canny == True)] = 200
        combined_result[np.where(draw_result == 100)] += 50

        for y in range(draw_result.shape[0]):
            for x in range(draw_result.shape[1]):
                if draw_result[y, x] == 100 and data_tested_canny[y, x] == True:
                    image_color_arr.putpixel((x, y), (255, 0, 0))

        f, images = plt.subplots(1, 2)
        images[0].imshow(combined_result, cmap='gray')
        images[0].set_title('Hyperbola ' + str(hyperbola_i) + '. a0= ' + str(result[2][hyperbola_i]) + ', ' +
                            'a1= ' + str(convexity_min+convexity_res*result[1][hyperbola_i]) + ', ' +
                            'a2= ' + str(result[0][hyperbola_i]) + ', score is: ' + str(result_score))
        images[1].imshow(image_color_arr, cmap='gray')
        plt.show()



    # transforming the data to the frequency space
    # fft_data = np.fft.fftshift(np.fft.fft2(gained_data))
    # data_max = abs(fft_data).max()
    # data_min = abs(fft_data).min()
    # data_mean = np.mean(abs(fft_data))
    # norm_fft_data = abs(fft_data)/(data_max-data_min)
    # lpf_fft_data = np.where(abs(fft_data) > 0.01*data_max, 1, 0)  # mask of the frequencies we want to keep
    # ifft_data = np.fft.ifft2(lpf_fft_data*fft_data)

    # plotting the data #
    f, images = plt.subplots(2, 2)
    plt.gray()
    images[0, 0].set_title('Synthietic Model Data')
    images[0, 0].imshow(resize(gained_sim, (sim_data.shape[0], sim_data.shape[0])))
    images[0, 0].set_xlabel(str(sim_data.shape[1] - 1) + ' A-scans')
    images[0, 0].set_ylabel(str(sim_data.shape[0] - 1) + ' Samples')
    images[0, 1].set_title('Original '+str(file_name)+' Recording With Gain Correction')
    images[0, 1].imshow(resize(gained_data, (int(0.5 * data.shape[1]), data.shape[1])))
    images[0, 1].set_xlabel('Horizontal distance: ' + str(num_distance) + '[m]')
    images[0, 1].set_ylabel('time window:' + str(num_time_window) + '[ns]')
    images[1, 0].set_title('Synthetic - Edges')
    images[1, 0].imshow(sim_canny)    # resize(sim_canny, (sim_canny.shape[0], sim_canny.shape[0])))
    images[1, 1].set_title('Real - Edges')
    images[1, 1].imshow(real_canny)    # resize(canny_edge, (int(0.5*data.shape[1]), data.shape[1])))
    plt.show()


if __name__ == '__main__':
    main()
