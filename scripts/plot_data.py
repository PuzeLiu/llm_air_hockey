import matplotlib.pyplot as plt
import sys
import numpy as np
import json
import os

data_dir = sys.argv[1]

def load(fn):
    with open(os.path.join(data_dir, fn), 'r') as f:
        data = json.load(f)['best_dist_to_goal']
    print("loaded", fn)
    return data

def data_filenames():
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            yield filename

def main():
    fig, ax = plt.subplots(layout='constrained')

    n = 0

    num_eps = []
    
    for filename in data_filenames():
        n += 1
        data = load(filename)
        
        num_eps.append(len(data))
        ax.plot(np.arange(len(data)), data)

    mede = np.median(num_eps)
    avge = np.mean(num_eps)
    stde = np.std(num_eps)
    maxe = np.max(num_eps)

    ax.grid()
    ax.set_xlabel('Episodes', fontsize=20)
    ax.set_ylabel('Distance (m)', fontsize=20)
    ax.set_title(f'Best dist2goal ({n} trails, avg eps {avge:.2f}±{stde:.2f}, median {mede}, max {maxe})', fontsize=24)

    plt.show()

if __name__ == '__main__':
    main()
    
