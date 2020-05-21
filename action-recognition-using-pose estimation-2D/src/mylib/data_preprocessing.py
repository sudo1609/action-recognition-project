import numpy as np

def pose_normalization(x):
    def retrain_only_body_joints(x_input):
        x0 = x_input.copy()
        x0 = x0[2:2+13*2]
        return x0

    def normalize(x_input):
        # Separate original data into x_list and y_list
        lx = []
        ly = []
        N = len(x_input)
        i = 0
        while i<N:
            lx.append(x_input[i])
            ly.append(x_input[i+1])
            i+=2
        lx = np.array(lx)
        ly = np.array(ly)

        # Get rid of undetected data (=0)
        non_zero_x = []
        non_zero_y = []
        for i in range(int(N/2)):
            if lx[i] != 0:
                non_zero_x.append(lx[i])
            if ly[i] != 0:
                non_zero_y.append(ly[i])
        if len(non_zero_x) == 0 or len(non_zero_y) == 0:
            return np.array([0] * N)

        # Normalization x/y data according to the bounding box
        origin_x = np.min(non_zero_x)
        origin_y = np.min(non_zero_y)
        len_x = np.max(non_zero_x) - np.min(non_zero_x)
        len_y = np.max(non_zero_y) - np.min(non_zero_y)
        x_new = []
        for i in range(int(N/2)):
            if (lx[i] + ly[i]) == 0:
                x_new.append(-1)
                x_new.append(-1)
            else:
                x_new.append((lx[i] - origin_x) / len_x)
                x_new.append((ly[i] - origin_y) / len_y)
        return x_new

    x_body_joints_xy = retrain_only_body_joints(x)
    x_body_joints_xy = normalize(x_body_joints_xy)
    return x_body_joints_xy
