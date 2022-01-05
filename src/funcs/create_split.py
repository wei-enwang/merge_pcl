import os
import random

r_val, r_test = 0.3, 0.1
seed = 32
work_dir = os.path.dirname("../data/obj/shapenet/")

random.seed(seed)

categories = sorted(os.listdir(work_dir))

for c_dir in categories:

    all_samples = [name for name in os.listdir(c_dir)
                if os.path.isdir(os.path.join(c_dir, name))]


    random.shuffle(all_samples)

    # Number of examples
    n_total = len(all_samples)
    n_val = int(r_val * n_total)
    n_test = int(r_test * n_total)

    if n_total < n_val + n_test:
        print('Error: too few training samples.')
        exit()

    n_train = n_total - n_val - n_test

    assert(n_train >= 0)

    # Select elements
    train_set = all_samples[:n_train]
    val_set = all_samples[n_train:n_train+n_val]
    test_set = all_samples[n_train+n_val:]

    # Write to file
    with open(os.path.join(c_dir, 'train.lst'), 'w') as f:
        f.write('\n'.join(train_set))

    with open(os.path.join(c_dir, 'val.lst'), 'w') as f:
        f.write('\n'.join(val_set))

    with open(os.path.join(c_dir, 'test.lst'), 'w') as f:
        f.write('\n'.join(test_set))
