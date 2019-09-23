import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sci
import h5py
from matplotlib import cm
import matplotlib.animation

DATA_PATH = "../../../DR/DR/ws/ratM/"
UNIT_CHOICE = 0
EXPORT_MOVIE = True

def main():
    f = h5py.File(DATA_PATH + "RatM_271118.mat") # Load matlab data file
    
    # Plot data from one trial to test the "clean_data" function
    x_raw = f['v']['pos']['Xpos']
    y_raw = f['v']['pos']['Ypos']
    unit_activity_raw = f['v']['spikes']['spikeHist'][UNIT_CHOICE]

    positions, speed, unit_activity = clean_data(x_raw, y_raw, unit_activity_raw)
    print(len(speed))
    np.save("../pca/M/speed.npy", np.asarray(speed))
    #plot_data(positions, unit_activity)


#------------------- Teleportation Removal Code ---------------##
def clean_data(x, y, unit_activity): 
    positions = np.hstack((np.transpose(x).astype(np.single), np.transpose(y).astype(np.single)))
    speed = []
    # return positions, speed, unit_activity
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

    return positions, speed, unit_activity.astype(np.int8)


def check_radius(prev_pos, pos, r):
    d = pos - prev_pos
    return np.inner(d, d) > r

##------------------ Plotting Code -----------------------##
def plot_data(positions, unit_activity):
    # history_length = 10
    cmap = cm.get_cmap('viridis', max(unit_activity) - 1)
    colors = cmap(unit_activity)#[matplotlib.colors.rgb2hex(x) for x in cmap(unit_activity)[:, :3]] # assign colors to every firing point
    colors[:, -1] = 0.15 # set alpha

    fig, ax = plt.subplots()
    ln = ax.scatter([0], [0], marker='o', linewidth=0, s=60)#, c=cmap(range(10)))

    plt.tick_params(
        axis='both',          # changes apply to both
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelbottom=False,
        labelleft=False) # labels along the bottom edge are off

    # Set up formatting for the movie files
    if EXPORT_MOVIE:
        writer = matplotlib.animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=12000)

    def init():
        ax.set_xlim(-30, 520)
        ax.set_ylim(65, 615)
        # Draw maze regions
        draw_maze()

        return ln,

    def update(frame):
        #Make a tuple or list of (x0,y0,c0,x1,y1,c1,x2....)
        temp = np.copy(colors[:frame])
        try:
            temp[frame - 1] = np.array([0,1,0, 1])
        except IndexError:
            pass
        ln.set_color(temp)
        # ln.set_sizes(np.transpose(unit_activity[:frame]))
        ln.set_offsets(positions[:frame])
        return ln,
      
    ani = matplotlib.animation.FuncAnimation(fig, update, 
            frames=(1200 if EXPORT_MOVIE else len(positions)), init_func=init, blit=True, interval=10)
    if EXPORT_MOVIE:
        ani.save("outputs/viridis.mp4", writer=writer, dpi=180)
        print("done")
    else: 
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

