import numpy as np
import matplotlib.pyplot as plt
import h5py

DATA_PATH = "/home/alex/Desktop/NPix_dataset/ratL/"
UNIT_CHOICE = 0 # Temporary until finished and clean over all units

def main():
    f = h5py.File(DATA_PATH + "RatL_241018.mat") # Load matlab data file
    x_raw = f['v']['pos']['Xpos']
    y_raw = f['v']['pos']['Ypos']
    unit_activity_raw = f['v']['spikes']['spikeHist'][UNIT_CHOICE] # debug check
    positions_new, speed, unit_activity = clean_data(x_raw, y_raw, unit_activity_raw)
    verify_trajectory(np.hstack((np.transpose(x_raw), np.transpose(y_raw))),positions_new)
    verify_across_all(x_raw, y_raw, f['v']['spikes']['spikeHist'])

def clean_data(x, y, unit_activity): 
    positions = np.hstack((np.transpose(x), np.transpose(y)))
    speed = []
    vel = np.array([0, 0])
    # Throw away first set of nans which 
    i = 0
    speed_max = 8
    r = 50
    while i < positions.shape[0]:
        # Delete leading Nan values
        if (i < 3) and (True in np.isnan(positions[i:i+3].flatten())):
            # print("before",  i, positions[i])
            positions = positions[1:, :]
            unit_activity = unit_activity[1:]
            # print("after", i, positions[i])
            continue
        
        ## Delete Nans / Teleports + interpolate linearly ##
        i = max(1, i)
        # print(positions[i-1:i+10]) 
        j = i 
        count = 1
        while ((True in np.isnan(positions[j]) or check_radius(positions[i - 1], positions[j], r))):
            count += 1
            j += 1
            r = 2 * ((speed_max * count) ** 2)
            # print(i,j, count, positions[j])
        if count > 1: # Some number of invalid values were encountered
            r = 50 # Reset valid point definition

            # Interpolate between first new valid point and last valid point
            vel_inbetween = (positions[j] - positions[i - 1]) / count 
            for k in range(i, j):
                positions[k] = positions[k - 1] + vel_inbetween
        ## 

        # Calculate speed
        velocity = positions[i] - positions[i-1]
        speed.append(np.inner(velocity, velocity))

        i += 1
    return positions, speed, unit_activity

def check_radius(prev_pos, pos, r):
    d = pos - prev_pos
    return np.inner(d, d) > r

def verify_trajectory(positions_old, positions_new):
    plot_range = len(positions_old)

    plt.subplot(1, 2, 1)
    plt.title("Old Positions")
    plt.scatter(positions_old[:plot_range, 0], positions_old[:plot_range, 1], s=0.1)
    plt.subplot(1, 2, 2)
    plt.title("New Positions")
    plt.scatter(positions_new[:plot_range, 0], positions_new[:plot_range, 1], s=0.1)
    plt.show()

if __name__ == "__main__":
    main()



## Old Teleportation Removal Code
    # # Note - unique auto sorts
    # Remove nan values
    # x = x[~np.isnan(x)]
    # y = y[~np.isnan(y)]
    # print(len(x))

    # # Calculate "velocity" (diff in subsequent positions)
    # vx = np.array([0] + [j - i for i, j in zip(x, x[1:])])
    # vy = np.array([0] + [j - i for i, j in zip(y, y[1:])])

    # # Filter out positions corresponding to abnormal velocities 
    # difference_threshold = 8
    # good_points = np.unique(np.hstack((np.where(np.abs(vx) < difference_threshold), np.where(np.abs(vy) < difference_threshold))))
    # x = x[good_points]
    # y = y[good_points]
    # unit_activity = unit_activity[good_points]

    # A better method with interpolation
    # while x < len bla

    # plt.hist(vx, bins=np.arange(-10, 10, 0.1))
    # plt.show()