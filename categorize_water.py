import numpy as np
import argparse
parser = argparse.ArgumentParser(prog='categorize_water.py',
                                 description='script that separates water molecules into surface and internal water and output them as pdbs',
                                 )
parser.add_argument('filename')

if __name__ == "__main__":
    args = parser.parse_args()
    print(args.filename)
    pass
