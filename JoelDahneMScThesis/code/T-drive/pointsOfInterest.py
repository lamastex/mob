import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

def homeLocation(r, grid):
    """Returns the most visited part of the grid as well as the timestamps
    when it was visited.
    """

    if r.size == 0:
        ## If no grids were visited return the zero grid with no times
        return (np.zeros(1, dtype=[("x", "int32"), ("y", "int32")])[0],
                np.array((0, 1), dtype="int32"))

    g = np.zeros(r.shape, dtype=[("x", "int32"), ("y", "int32")])

    g["x"] = np.floor(r["x"]/grid[0])
    g["y"] = np.floor(r["y"]/grid[0])

    u, i, c = np.unique(g, return_counts=True, return_inverse=True)

    home = u[c.argmax()]

    times = r["t"][i == c.argmax()]

    return (home, times)

def homeLocations(R, grid, all=True):
    """Returns a dictionary containing the most visited part of the grid
    as well as the timestamps when it was visited for each trajectory.

    If the all flag is set to false only counts home locations if more
    than 5% of the time was spent there. For the trajectories with no
    determined home location nothing is inserted into the dictionary
    in this case.

    """

    homes = dict()

    for i, r in R.items():
        home = homeLocation(r, grid)
        if all or home[1].shape[0] > 0.05*r.shape[0]:
            homes[i] = homeLocation(r, grid)

    return homes

def homeDistribution(R, homes, grid):
    """Returns the percantage of measurements occuring at the computed
    home location for each hour of the day. Only considers
    trajectories for which a home location is determined.
    """
    ## Compute counts per hour for home location
    rows = sum(t.shape[0] for i, (_, t) in homes.items())

    hours = np.zeros(rows, dtype="int32")

    row = 0

    for i, (_, t) in homes.items():
        hours[row:row + t.shape[0]] = list(map(lambda x:
                                               datetime.fromtimestamp(x).hour,
                                               t))

        row = row + t.shape[0]

    _, homePerHour = np.unique(hours, return_counts=True)

    ## Compute counts per hour for all measurements
    rows = sum(R[i].shape[0] for i in homes)

    hours = np.zeros(rows, dtype="int32")

    row = 0

    for i in homes:
        r = R[i]

        hours[row:row + r.shape[0]] = list(map(lambda x: datetime.fromtimestamp(x).hour, r["t"]))

        row = row + r.shape[0]

    _, totalPerHour = np.unique(hours, return_counts=True)

    return homePerHour/totalPerHour

if __name__ == "__main__":
    grid = (np.float64(0.001), np.float64(60))
    originalFile = "R-original.file"
    swappedFile = "R-swapped.file"

    ## Load data
    print("Loading co-trajectories")

    with open(originalFile, "rb") as f:
        originalR = pickle.load(f)

    with open(swappedFile, "rb") as f:
        swappedR = pickle.load(f)

    ## Compute home locations and distributions for the original
    ## co-trajectory
    print("Computing frequency histogram of home locations for the original co-trajectory")

    originalHomes = homeLocations(originalR, grid)

    originalDist = homeDistribution(originalR, originalHomes, grid)

    ## Plot the distributions
    plotFile = "HomeHour.pdf"
    print("Plotted frequencey histogram to the file " + plotFile)

    plt.bar(range(24), originalDist, align="edge")

    plt.title("Frequency histogram of hours at deduced home locations")
    plt.xlabel("Hour")
    plt.ylabel("Relative frequency")

    plt.savefig(plotFile)

    ## Compare number of "correctly" guessed home location before and
    ## after swapping. This is done by computing home locations for
    ## the original co-trajectory, only keeping those who are home at
    ## least 5% of the time, and comparing this to the computed home
    ## locations for the swapped co-trajectory, again only keeping
    ## those who are home at least 5% of the time.
    print("Computing correctly guessed home locations before and after swapping")

    originalHomes = homeLocations(originalR, grid, all=False)

    swappedHomes = homeLocations(swappedR, grid, all=False)

    correct = 0

    for i in originalR:
        if i in originalHomes and i in swappedHomes and originalHomes[i][0] == swappedHomes[i][0]:
            correct = correct + 1

    print("Guessed home locations for the original co-trajectory visited by")
    print("more than 5% of the measurements: " + str(len(originalHomes)) + "\n")

    print("Guessed home locations for the swapped co-trajectory visited by")
    print("more than 5% of the measurements: " + str(len(swappedHomes)) + "\n")

    print("Number of these that coincide: " + str(correct))
