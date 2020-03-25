from dataset import PixelSetData_preloaded, PixelSetData
import argparse
import time

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--npixel', type=int, default=64)
    args = parser.parse_args()


    dt_normal = PixelSetData(folder=args.dataset, labels='label_19class', npixel=args.npixel)
    dt_ram = PixelSetData_preloaded(folder=args.dataset, labels='label_19class', npixel=args.npixel)


    t0 = time.time()
    for x, y in dt_normal:
        x[0].mean()
    t1 = time.time()

    print('Total time with normal dataset: {}'.format(t1 - t0))

    t0 = time.time()
    for x, y in dt_ram:
        x[0].mean()
    t1 = time.time()

    print('Total time with dataset in RAM: {}'.format(t1 - t0))