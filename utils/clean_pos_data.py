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