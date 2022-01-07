import argparse
import os
import glob
import random

# parser = argparse.ArgumentParser(
#     description='Split data into train, test and validation sets.')
# parser.add_argument('in_folder', type=str,
#                     help='Input folder where data is stored.')

# parser_nval = parser.add_mutually_exclusive_group(required=True)
# parser_nval.add_argument('--n_val', type=int,
#                          help='Size of validation set.')
# parser_nval.add_argument('--r_val', type=float,
#                          help='Relative size of validation set.')

# parser_ntest = parser.add_mutually_exclusive_group(required=True)
# parser_ntest.add_argument('--n_test', type=int,
#                           help='Size of test set.')
# parser_ntest.add_argument('--r_test', type=float,
#                           help='Relative size of test set.')

# parser.add_argument('--shuffle', action='store_true')
# parser.add_argument('--seed', type=int, default=4)

# args = parser.parse_args()

"""make sure all files are in the SAME format under the directory"""

random.seed(32)
r_val = 0.3
r_test = 0.1

in_folder = "../data/pcl/"

for p in os.listdir(in_folder):
    work_dir = os.path.join(in_folder, p)
    if os.path.isdir(work_dir):
        
        samples = []
        for filetype in ["*.off", "*.npz", "*.obj"]:
            samples.extend(glob.glob(work_dir+filetype))
        random.shuffle(samples)

        # Number of examples
        n_total = len(samples)

        n_val = int(r_val * n_total)
        n_test = int(r_test * n_total)

        if n_total < n_val + n_test:
            print('Error: too few training samples.')
            exit()

        n_train = n_total - n_val - n_test

        assert(n_train >= 0)

        # Select elements
        train_set = samples[:n_train]
        val_set = samples[n_train:n_train+n_val]
        test_set = samples[n_train+n_val:]

        # Write to file
        with open(os.path.join(work_dir, 'train.lst'), 'w') as f:
            f.write('\n'.join(train_set))

        with open(os.path.join(work_dir, 'val.lst'), 'w') as f:
            f.write('\n'.join(val_set))

        with open(os.path.join(work_dir, 'test.lst'), 'w') as f:
            f.write('\n'.join(test_set))
