import numpy as np
import matplotlib.pyplot as plt
import random

## Trajectory r = [((x, y), t)]
## Individuals i = int
## Co-trajectory R = dict(individual, trajectory)
## Swap s = (i_1, i_2, t_1, t_2)

def swapTrajectories(r1, r2, t1, t2):
    """Swap the two trajectories r1 and r2 at time (t1, t2)."""
    r1p = ([(l, t) for (l, t) in r1 if t < t1]
           + [(l, t) for (l, t) in r2 if t >= t2])
    r2p = ([(l, t) for (l, t) in r2 if t < t2]
           + [(l, t) for (l, t) in r1 if t >= t1])
    return r1p, r2p

def swap(R, s):
    """Perform the swap given by s on the co-trajectory R."""
    Rp = R.copy()
    Rp[s[0]], Rp[s[1]] = swapTrajectories(R[s[0]], R[s[1]], s[2], s[3])
    return Rp

def updateSwaps(S, s):
    """Update the set of swaps S so that they are all valid after
    performing the swap s.
    """
    Sp = set()
    for sp in S:
        # Check if the first id is equal to one of the swapped ones
        # and the time is later than the swapped time, if so swap it
        if (sp[0] == s[0] and sp[2] >= s[2]):
            sp = (s[1], sp[1], sp[2], sp[3])
        elif (sp[0] == s[1] and sp[2] >= s[3]):
            sp = (s[0], sp[1], sp[2], sp[3])

        # Check if the second id is equal to one of the swapped ones
        # and the time is later than the swapped time, if so swap it
        if (sp[1] == s[0] and sp[3] >= s[2]):
            sp = (sp[0], s[1], sp[2], sp[3])
        elif (sp[1] == s[1] and sp[3] >= s[3]):
            sp = (sp[0], s[0], sp[2], sp[3])

        Sp.add(sp)

    return Sp

def dist(x1, x2):
    """Returns the distance between x1 and x2."""
    return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)

## Hard coded values for chi and tau
chi = 0.1
tau = 0.1

def similar(m1, m2):
    """Returns true if the measurements m1 and m2 are similar."""
    if m1 == m2:
        return False

    if dist(m1[0], m2[0]) <= chi and abs(m1[1] - m2[1]) <= tau:
        return True

    return False

def swapsTrajectories(r1, r2, i1=None, i2=None):
    """Return the set of all valid swaps between the two trajectories r1
    and r2. If two ids are given, add these two to the swaps.
    """
    S = set()
    for m1 in r1:
        for m2 in r2:
            if similar(m1, m2):
                if i1 == None and i2 == None:
                    S.add((m1[1], m2[1]))
                else:
                    S.add((i1, i2, m1[1], m2[1]))
    return S

def swaps(R):
    """Return the set of all valid swaps on R."""
    S = set()
    for i1 in R:
        for i2 in R:
            if i1 <= i2:
                S.update(swapsTrajectories(R[i1], R[i2], i1, i2))

    return S

def SwapMob(R):
    S = swaps(R)
    while len(S) > 0:
        ## Pick a random swap to perform
        s = S.pop()
        ## Check so that it is not a swap with itself
        if s[0] != s[1]:
            ## Randomly choose to apply the swap or not
            if random.choice([True, False]):
                R = swap(R, s)
                S = updateSwaps(S, s)

    return R

## Methods for plotting co-trajectories
def plotTrajectory(r):
    r = sorted(r, key=lambda m: m[1])
    plt.plot(list(map(lambda m: m[0][0], r)),
             list(map(lambda m: m[0][1], r)), "o-")

def plotCoTrajectory(R):
    for i in R:
        plotTrajectory(R[i])
