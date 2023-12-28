# remove *.pth files in given dir, with undevisable of number 25, the format of the file name is like: epoch_30.pth

import argparse
import glob
import os
import shutil
import sys

from send2trash import send2trash


def parse_args():
    parser = argparse.ArgumentParser(
        description="remove *.pth files in given dir, with undevisable of number 25, the format of the file name is like: epoch_30.pth"
    )
    parser.add_argument("dir", help="dir to remove *.pth files")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dir = args.dir
    if not os.path.isdir(dir):
        print(f"{dir} is not a dir")
        sys.exit(1)
    for pth in glob.glob(os.path.join(dir, "*.pth")):
        basename = os.path.basename(pth)
        if basename.startswith("epoch_"):
            epoch = int(basename.split("_")[1].split(".")[0])
            if epoch % 25 != 0:
                print(f"remove {pth}")
                send2trash(pth)

        # else:
        #     print(f"remove {pth}")
        #     os.remove(pth)


if __name__ == "__main__":
    main()
