

from pathlib import Path
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('folder', 
                    help="supply a folder path to be split up. if not folder, method won't do anything")

args = parser.parse_args()

MAX_FOLDER_SIZE = 2000


def split_files(folder: Path):

    assert folder.is_dir()
    num_dirs = 1
    curr_size = 0

    new_subfolder_path = folder.parent / f"{folder.as_posix()}_{num_dirs}"
    new_subfolder_path.mkdir(exist_ok=True)
    for file in folder.iterdir():
        if curr_size > MAX_FOLDER_SIZE:
            num_dirs += 1
            new_subfolder_path = folder.parent / f"{folder.as_posix()}_{num_dirs}"
            new_subfolder_path.mkdir(exist_ok=True)
            curr_size = 0
        else:
            curr_size += 1
        file.rename(new_subfolder_path / file.name)

if __name__ == "__main__":
    split_files(Path(args.folder))

