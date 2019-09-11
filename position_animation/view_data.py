import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sci
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
DATA_PATH = "/home/alex/Desktop/NPix_dataset/ratL/"

def main():
    f = h5py.File(DATA_PATH + "RatL_241018.mat") # Load matlab data file
    
    # Plot data from one trial to test the "clean_data" function
    x_raw = f['v']['pos']['Xpos'][0]
    y_raw = f['v']['pos']['Ypos'][0]

    x, y, xy, vy = clean_data(x_raw, y_raw)
    plot_data(x, y, vy, vy)

def clean_data(x, y): 
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    # Calculate "velocity" (diff in subsequent positions)
    vx = np.array([0] + [j - i for i, j in zip(x, x[1:])])
    vy = np.array([0] + [j - i for i, j in zip(y, y[1:])])

    # Filter out positions corresponding to abnormal velocities 
    difference_threshold = 10
    good_points = np.unique(np.hstack((np.where(np.abs(vx) < difference_threshold), np.where(np.abs(vy) < difference_threshold))))
    x = x[good_points]
    y = y[good_points]

    # plt.hist(vx, bins=np.arange(-10, 10, 0.1))
    # plt.show()
    return x, y, vx, vy


    

##------------------ Plotting Code -----------------------##
def plot_data(x, y, vx, vy):
    fig, ax = plt.subplots()
    history_length = 10
    cmap = plt.cm.Spectral
    ln, = plt.plot([0]*10, [0]*10, 'o')#, c=cmap(range(10)))
   
    # Draw maze regions
    draw_maze()

    def init():
        ax.set_xlim(0, 550)
        ax.set_ylim(0, 600)
        return ln,

    def update(frame):
        if frame < history_length:
            ln.set_data(x[0:frame], y[0:frame])
        else:
            ln.set_data(x[frame - history_length:frame], y[frame - history_length:frame])
        return ln,
        
    ani = FuncAnimation(fig, update, frames=range(len(x)),
                    init_func=init, blit=True, interval=30)

    plt.show()

def draw_maze():
    import scipy.io as sci
    boundary_def = sci.loadmat("/home/alex/Desktop/ndc/neuropixel/position_animation/coordinatesData.mat")
    print(defn)
    for box in defn:


##---------------------------------------------------------##

if __name__ == "__main__":
    print("--- Starting Animation ---")
    main()

