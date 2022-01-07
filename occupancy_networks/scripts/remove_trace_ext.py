import os
import glob

files = glob.glob('../data/**/*.off.*')
for f in files:
    os.rename(f, f[:-8]+f[-4:])