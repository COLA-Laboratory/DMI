import sys
import os

print('Python %s on %s at %s' % (sys.version, sys.platform, os.getcwd()))
sys.path.extend(['./'])

from algorithms.DmiEA import DmiEA
from algorithms.FullSearch import FullSearch
from algorithms.MGD import MGD
from algorithms.ParEGO import ParEGO
from algorithms.MoeadEGO import MoeadEGO
from utils.get_arguments import get_args


def main():
    args, _ = get_args()
    if args.algorithm == 'MGD':
        MGD(args).run()
    elif args.algorithm == 'ParEGO':
        ParEGO(args).run()
    elif args.algorithm == 'MoeadEGO':
        MoeadEGO(args).run()
    elif args.algorithm == 'FullSearch':
        FullSearch(args).run()
    elif args.algorithm == 'DMI':
        DmiEA(args).run()
    else:
        print(f'Unknown algorithm{args.algorithm}')


if __name__ == '__main__':
    main()
