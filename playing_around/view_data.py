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

    x, y = clean_data(x_raw, y_raw)
    plot_data(x, y)

def clean_data(x, y): 
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    return x, y



##------------------ Plotting Code -----------------------##
def plot_data(x, y):
    fig, ax = plt.subplots()
    history_length = 10
    cmap = plt.cm.Spectral
    ln, = plt.plot([0]*10, [0]*10, 'o')#, c=cmap(range(10)))
    plt.hlines(50, xmin=0, xmax=45)

    def init():
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 600)
        return ln,

    def update(frame):
        if frame < history_length:
            ln.set_data(x[0:frame], y[0:frame])
        else:
            ln.set_data(x[frame - history_length:frame], y[frame - history_length:frame])
        return ln,

    ani = FuncAnimation(fig, update, frames=range(len(x)),
                    init_func=init, blit=True, interval=200)

    plt.show()
##---------------------------------------------------------##

if __name__ == "__main__":
    print("--- Starting Animation ---")
    main()

