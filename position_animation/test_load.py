import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
sys.path.append(os.path.dirname(os.path.realpath(__file__))[:-18])
from classes.DataGetter import DataGetter

path = "/home/alex/Desktop/NPix_dataset/ratL/RatL_241018.mat"
posns, speed, unit_firing = DataGetter(path).get_data()

bin_size = 15
maze_extent_x = 450
maze_extent_y = 480
num_units = min(136, 20)
visit_record = np.zeros((int(maze_extent_x/bin_size), int(maze_extent_y/bin_size)))
offset = np.array([-20, 110])
INDIVIDUAL_PLOT = True

for i in range(len(posns)):
    coord = ((posns[i]-offset)/bin_size).astype(np.int8)
    visit_record[coord[0], coord[1]] += 1

if INDIVIDUAL_PLOT:
    dim = int(np.ceil(np.sqrt(num_units)))
    nrow = dim - 1
    ncol = dim
    fig = plt.figure(figsize=(ncol+1, nrow+1)) 
    gs = gridspec.GridSpec(nrow, ncol, wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    # gs = gridspec.GridSpec(nrow, ncol,
    #         wspace=0.0, hspace=0.0, vspace=0.0,
    #         top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
    #         left=0.5/(ncol+1), right=1-0.5/(ncol+1))
    unit_id = 0
    for plot_x in range(nrow):
        for plot_y in range(ncol):
            ax= plt.subplot(gs[plot_x,plot_y])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if unit_id >= num_units: continue
            arr = np.zeros((int(maze_extent_x/bin_size), int(maze_extent_y/bin_size)))
            for i in range(len(posns)):
                coord = ((posns[i]-offset)/bin_size).astype(np.int8)
                arr[coord[0], coord[1]] += unit_firing[unit_id][i]
            unit_id += 1

            ax.imshow(arr/visit_record)

                # print(posns[i], (posns[i]/bin_size).astype(np.int8))
    plt.show()
else:
    arr = np.zeros((int(maze_extent_x/bin_size), int(maze_extent_y/bin_size)))
    for i in range(len(posns)):
            coord = ((posns[i]-offset)/bin_size).astype(np.int8)
            arr[coord[0], coord[1]] += np.sum(unit_firing[:, i])
    # plt.imshow(arr, cmap="coolwarm")
    plt.imshow(arr / visit_record, cmap="viridis")
    plt.show()
