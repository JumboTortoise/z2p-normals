"""
a short script that removes cached zbuffers form the dataset
"""
from pathlib import Path
import argparse

def main(args):
    if not args.root.is_dir():
        raise Exception("the provided path is not a directory")
        return


    shapes = [shape for shape in args.root.resolve().iterdir() if shape.is_dir()]    
    
    for shape in shapes:
        print("cleaning",str(shape))
        rotations = [rot for rot in shape.iterdir() if rot.is_dir()]
        for rotation in rotations:
            to_delete = lambda file: file.is_file() and file.suffix == '.npy' and 'cache' in file.stem
            datafiles = [file for file in rotation.iterdir() if to_delete(file)]    
            for f in datafiles:
                f.unlink()
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root",type=Path,help='the root dataset directory')
    args = parser.parse_args()

    main(args)
