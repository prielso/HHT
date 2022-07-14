import numpy as np
import h5py
# from ctypes import string_at
# from gsfpy3_08 import open_gsf
# from gsfpy3_08.enums import RecordType
from skimage.filters import gaussian


# with open_gsf("D://GPR_local/new_recordings/Profile49.gsf") as gsf_file:
#     # Note - file is closed automatically upon exiting 'with' block
#     _, record = gsf_file.read(RecordType.GSF_RECORD_COMMENT)
#
#     # Note use of ctypes.string_at() to access POINTER(c_char) contents of
#     # c_gsfComment.comment field.
#     print(string_at(record.comment.comment))


def get_output_data(filename, rxnumber, rxcomponent):
    """Gets B-scan output data from a synthetic model.
    Args:
        filename (string): Filename (including path) of output file.
        rxnumber (int): Receiver output number.
        rxcomponent (str): Receiver output field/current component.
    Returns:
        outputdata (array): Array of A-scans, i.e. B-scan data.
        dt (float): Temporal resolution of the model.
    """
    # Open output file and read some attributes
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    dt = f.attrs['dt']
    # Check there are any receivers
    # if nrx == 0:
    #     raise CmdInputError('No receivers found in {}'.format(filename))
    path = '/rxs/rx' + str(rxnumber) + '/'
    availableoutputs = list(f[path].keys())
    # Check if requested output is in file
    # if rxcomponent not in availableoutputs:
    #     raise CmdInputError('{} output requested to plot, but the available output for receiver 1 is
    #     {}'.format(rxcomponent, ', '.join(availableoutputs)))
    outputdata = f[path + '/' + rxcomponent]
    outputdata = np.array(outputdata)
    f.close()
    return outputdata, dt


# this function correct the signal decay during time, adding time dependant gain, the longer the signal is going the
# more the signal is decreasing, usually the signal is decaying exponentially so an exponent gain is given.
def decay_correction(data, std, amp, air_gap):
    gained_data = data
    for i in list(range(air_gap, data.shape[0])):
        gained_data[i, :] = data[i, :] * amp * np.exp(i/std)
    return gained_data


def time_radius_correction(data):
    pixel_cnt = np.zeros(data.shape)
    new_data = np.zeros(data.shape)
    for row in range(data.shape[0]):   # row is the Time Dimension.
        for col in range(data.shape[1]):  # col is the number of the Horizontal location on the surface.
            for side_col in range(row+1):
                row_in_radius = int(np.sqrt(row ** 2 - side_col ** 2))
                if (col-side_col) in range(data.shape[1]):
                    new_data[row_in_radius, col - side_col] += data[row, col]
                    pixel_cnt[row_in_radius, col - side_col] += 1
                if ((col+side_col) in range(data.shape[1])) and (side_col != 0):
                    new_data[row_in_radius, col + side_col] += data[row, col]
                    pixel_cnt[row_in_radius, col + side_col] += 1
    return new_data, pixel_cnt


def hyperbola_draw(a0, a1, a2, dim):
    drawing = np.zeros(dim)
    drawing_dy = np.zeros(dim)
    for x in range(dim[1]):
        y = a0+a1*(x-a2)**2
        dy = 2*a1*(x - a2)
        if int(y) < dim[0]:
            drawing[int(y), x] = 100
            drawing_dy[int(y), x] = dy
    # drawing = gaussian(drawing, sigma=2)
    return drawing, drawing_dy




