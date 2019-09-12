""" Provide class with path to your .mat data file,
    returns cleaned / non-cleaned datasets """

import numpy as np
import matplotlib.pyplot as plt
import h5py

class DataGetter:
    def __init__(self, path, unit_choice=-1):
        self.f = h5py.File(path) # Load matlab data file

    def get_data(self, clean=True, unit_choice=0):
        if not clean:
            return self.f['v']
        else:
            # TODO Unit choice -1 ==> All units
            x_raw = self.f['v']['pos']['Xpos']
            y_raw = self.f['v']['pos']['Ypos']
            unit_activity_raw = self.f['v']['spikes']['spikeHist'][unit_choice] 
            return self.clean_data(x_raw, y_raw, unit_activity_raw)
        

    def clean_data(self, x, y, unit_activity): 
        def check_radius(prev_pos, pos, r):
            d = pos - prev_pos
            return np.inner(d, d) > r

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