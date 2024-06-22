import numpy as np
import glob

if __name__ == "__main__":
    suffix = 'train'
    max_list, min_list = [], []
    for file in glob.glob(f'data/{suffix}/processed/*.pkl'):
        data = np.load(file, allow_pickle=True)['y']
        max_list.append(np.max(data))
        min_list.append(np.min(data))

    print(np.max(max_list), np.min(min_list))

    with open(f'data/stats.txt', 'w') as f:
        f.write(f'{np.max(max_list)}\n')
        f.write(f'{np.min(min_list)}')