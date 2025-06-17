import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from pathlib import Path

recordings = [
    "wall1",
    "wall2",
    "wall3"
]
algorithms =["BeamformerBase","BeamformerOrth","BeamformerCleansc","BeamformerCMF"]


# Porównanie dla każdego algorytmu
for algo in algorithms:
    fig, ax = plt.subplots()
    ax.set_title(f"Cumulative error distribution ({algo})")

    for rec in recordings:
        dist_file = Path(f"./dist/{rec}_{algo}_dist_focuspoints.npy")
        dist = np.load(dist_file)
        hist, bins = np.histogram(dist, bins=100, range=(0, 100))
        ax.plot(bins[:-1], hist.cumsum() / hist.sum(), label=f"{rec}")

    ax.set_xlabel("Distance [px]")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.grid()
    ax.legend(loc='lower right')
    plt.savefig(f"./ced/{algo}_ced.svg", format='svg')
    plt.close()

# Porównanie wszystkich algorytmów dla jednego nagrania
# wall1
fig, ax = plt.subplots()
ax.set_title("Cumulative error distribution (all algorithms - wall1)")
for algo in algorithms:
    dist_file = Path(f"./dist/wall1_{algo}_dist_focuspoints.npy")
    try:
        dist = np.load(dist_file)
    except FileNotFoundError:
        print(f"File {dist_file} does not exist. Skipping {algo}.")
        continue
    hist, bins = np.histogram(dist, bins=100, range=(0, 100))
    ax.plot(bins[:-1], hist.cumsum() / hist.sum(), label=algo)

ax.set_xlabel("Distance [px]")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.grid()
ax.legend(loc='lower right')
plt.savefig("./ced/wall1_ced_all_.svg", format='svg')
plt.close()

#wall2
fig, ax = plt.subplots()
ax.set_title("Cumulative error distribution (all algorithms - wall2)")
for algo in algorithms:
    dist_file = Path(f"./dist/wall2_{algo}_dist_focuspoints.npy")
    try:
        dist = np.load(dist_file)
    except FileNotFoundError:
        print(f"File {dist_file} does not exist. Skipping {algo}.")
        continue
    hist, bins = np.histogram(dist, bins=100, range=(0, 100))
    ax.plot(bins[:-1], hist.cumsum() / hist.sum(), label=algo)

ax.set_xlabel("Distance [px]")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.grid()
ax.legend(loc='lower right')
plt.savefig("./ced/wall2_ced_all_.svg", format='svg')
plt.close()

#wall3
fig, ax = plt.subplots()
ax.set_title("Cumulative error distribution (all algorithms - wall3)")
for algo in algorithms:
    dist_file = Path(f"./dist/wall3_{algo}_dist_focuspoints.npy")
    try:
        dist = np.load(dist_file)
    except FileNotFoundError:
        print(f"File {dist_file} does not exist. Skipping {algo}.")
        continue
    hist, bins = np.histogram(dist, bins=100, range=(0, 100))
    ax.plot(bins[:-1], hist.cumsum() / hist.sum(), label=algo)

ax.set_xlabel("Distance [px]")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.grid()
ax.legend(loc='lower right')
plt.savefig("./ced/wall3_ced_all_.svg", format='svg')
plt.close()
