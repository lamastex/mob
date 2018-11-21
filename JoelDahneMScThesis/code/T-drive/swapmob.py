import os
import csv
import pickle
import numpy as np
from datetime import datetime

## Datatypes

## Data type for the numpy array representing trajectories as part of
## a cotrajectory. The data contains location and time data, the
## location is given as two 64 bits floating point numbers and the
## time is in unix time given by a 32 bit integer.
Rtype = np.dtype([("x", "float64"), ("y", "float64"), ("t", "int32")])

## Data type for the numpy array consisting of measurements. It
## contains the same kind of data as Rtype with the addition of an id
## represented by a 32 bit integer.
Mtype = np.dtype([("x", "float64"), ("y", "float64"), ("t", "int32"),
                  ("id", "int32")])

## Data type for the numpy array consisting of grid values for
## measurements. It contains the same kind of data as Mtype with the
## difference that the location is now given by two 32 bit integers.
Gtype = np.dtype([("x", "int32"), ("y", "int32"), ("t", "int32"),
                  ("id", "int32")])

## Constants
datestr = "%Y-%m-%d %H:%M:%S"
path = "/home/urathai/Datasets/public/T-drive Taxi Trajectories/release/taxi_log_2008_by_id/"

def parseMeasurement(x, y, t):
    """Parse a measurement given by the strings x, y and t. Convertes x
    and y to floating points and t to unix time.
    """
    return (np.float64(x), np.float64(y),
            np.int32(datetime.strptime(t, datestr).timestamp()))

def parse(path):
    """Given a path to a directory containing taxi driver data as
    specified in the Beijing T-drive data set returns a co-trajectory
    consisting of the trajectories of all the taxis. The co-trajectory
    is given as a dictionary with the ids as keys and the trajectories
    given by numpy arrays of type Rtype.

    Every file should correspond to one taxi and the id is given by
    the filename after removing the .txt part.

    """
    files = os.listdir(path)

    R = dict()

    for fName in sorted(files):
        numLines = sum(1 for line in open(path + fName))

        r = np.zeros(numLines, dtype=Rtype)
        with open(path + fName) as f:
            print("Opened " + fName)

            read = csv.reader(f)

            for i, (_, t, x, y) in enumerate(read):
                r[i] = parseMeasurement(x, y, t)

            R[int(fName.split(".")[0])] = r

    return R

def toMeasurements(R):
    """Given a co-trajectory R returns a numpy array with the type of
    Mtype consisting of all the measurements.

    """
    rows = sum(R[i].shape[0] for i in R)

    M = np.zeros(rows, dtype=Mtype)

    row = 0

    for i, r in R.items():
        M["x"][row:row + r.shape[0]] = r["x"]
        M["y"][row:row + r.shape[0]] = r["y"]
        M["t"][row:row + r.shape[0]] = r["t"]
        M["id"][row:row + r.shape[0]] = i

        row = row + r.shape[0]

    return M

def toGrid(M, grid):
    """Given an array of measurements (of type Mtype) converts all
    measurements to a grid representation and returns an array of type
    Gtype.

    """
    G = np.zeros(M.shape, dtype=Mtype)

    G["x"] = np.floor(M["x"]/grid[0])
    G["y"] = np.floor(M["y"]/grid[0])
    G["t"] = np.floor(M["t"]/grid[1])
    G["id"] = M["id"]

    return G

def swaps(G, grid):
    """Returns all possible swaps, the returned list is ordered by time in
    descending order.

    """

    G = np.unique(G)

    print("Uniqued")

    G.sort(order=["x", "y", "t"])

    print("Sorted!")

    _, i, c = np.unique(G[["x", "y", "t"]], return_index=True, return_counts=True, axis=0)

    i = i[c > 1]
    c = c[c > 1]

    print("Collected")

    S = [((G["t"][j] + 1)*grid[1], G["id"][j:(j + d)]) for (j, d) in zip(i, c)]

    S.sort(key=lambda x: x[0], reverse=True)

    return S

def swap(R, s, p):
    """Perform the swap s according to permutation p on the cotrajectory
    R. The swap is done in place.

    """
    tails = dict()

    for (i, j) in zip(s[1], p):
        tails[i] = R[j][R[j]["t"] >= s[0]]

    for i in tails:
        R[i] = np.concatenate([R[i][R[i]["t"] < s[0]], tails[i]])

def swapmob(R, S):
    """Perform all the swaps in S on the co-trajectory R, choosing a
    random permutation for each one.

    """
    for i, s in enumerate(S):
        p = np.random.permutation(s[1])

        swap(R, s, p)

        if i % 10000 == 0:
            print(i)

if __name__ == "__main__":
    grid = (np.float64(0.001), np.float64(60))
    originalFile = "R-original.file"
    swappedFile = "R-swapped.file"

    R = parse(path)

    with open(originalFile, "wb") as f:
        pickle.dump(R, f)

    print("Saved parsed data in " + originalFile)

    G = toGrid(toMeasurements(R), grid)

    print("Mapped the data to the grid")

    S = swaps(G, grid)

    print("Computed swaps")

    swapmob(R, S)

    print("Performed the swaps")

    with open(swappedFile, "wb") as f:
        pickle.dump(R, f)

    print("Saved swapped data in " + swappedFile)
