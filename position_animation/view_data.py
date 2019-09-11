import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sci
import h5py
from matplotlib import cm
from matplotlib.animation import FuncAnimation
DATA_PATH = "/home/alex/Desktop/NPix_dataset/ratL/"
UNIT_CHOICE = 0
def main():
    f = h5py.File(DATA_PATH + "RatL_241018.mat") # Load matlab data file
    
    # Plot data from one trial to test the "clean_data" function
    x_raw = f['v']['pos']['Xpos'][0]
    y_raw = f['v']['pos']['Ypos'][0]
    unit_activity_raw = f['v']['spikes']['spikeHist'][UNIT_CHOICE]

    x, y, xy, vy, unit_activity = clean_data(x_raw, y_raw, unit_activity_raw)
    plot_data(x, y, vy, vy, unit_activity)

def clean_data(x, y, unit_activity): 
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    # Calculate "velocity" (diff in subsequent positions)
    vx = np.array([0] + [j - i for i, j in zip(x, x[1:])])
    vy = np.array([0] + [j - i for i, j in zip(y, y[1:])])

    # Filter out positions corresponding to abnormal velocities 
    difference_threshold = 8
    good_points = np.unique(np.hstack((np.where(np.abs(vx) < difference_threshold), np.where(np.abs(vy) < difference_threshold))))
    x = x[good_points]
    y = y[good_points]
    unit_activity = unit_activity[good_points]

    # A better method with interpolation
    # while x < len bla

    # plt.hist(vx, bins=np.arange(-10, 10, 0.1))
    # plt.show()
    return x, y, vx, vy, unit_activity

##------------------ Plotting Code -----------------------##
def plot_data(x, y, vx, vy, unit_activity):
    fig, ax = plt.subplots()
    history_length = 10
    cmap = cm.get_cmap('viridis', 8)
    ln, = plt.plot([0]*10, [0]*10, 'o')#, c=cmap(range(10)))
    plt.tick_params(
        axis='both',          # changes apply to both
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False) # labels along the bottom edge are off

    def init():
        ax.set_xlim(-30, 520)
        ax.set_ylim(65, 615)
        # Draw maze regions
        draw_maze()

        return ln,

    def update(frame):
        if frame < history_length:
            ln.set_data(x[0:frame], y[0:frame])
        else:
            ln.set_data(x[frame - history_length:frame], y[frame - history_length:frame])
        ln.set_color(cmap(unit_activity[frame]))
        return ln,
        
    ani = FuncAnimation(fig, update, frames=range(len(x)),
                    init_func=init, blit=True, interval=50)

    plt.show()

def draw_maze():
    import scipy.io as sci
    boundary_def = sci.loadmat("/home/alex/Desktop/ndc/neuropixel/position_animation/coordinatesData.mat")

    for box in boundary_def['coordinatesData'][0][0]:
        try:
            plt.hlines(xmin=box[0][0], xmax=box[2][0], y=box[0][1])
            plt.hlines(xmin=box[1][0], xmax=box[3][0], y=box[1][1])
            plt.vlines(ymin = box[1][1], ymax = box[0][1], x = box[0][0])
            plt.vlines(ymin = box[3][1], ymax = box[2][1], x = box[3][0])
        except IndexError:
            continue

##---------------------------------------------------------##

if __name__ == "__main__":
    print("--- Starting Animation ---")
    main()

