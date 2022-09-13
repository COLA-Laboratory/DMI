from argparse import ArgumentParser
import numpy as np

#args = Namespace(  var_bound=None,
#                     n_init=n_init, n_iter=n_iter, output=output, batch=10, selection_type='DHVI',
#                     algorithm_name='DimMOEAD',ea_type='moead')

def get_args(parser=None):
    '''
        args = Namespace(
                    seed=seed,
                    n_var=n_var,
                    n_init0=n_init0s[bo_type],n_iter0=n_iter0s[bo_type],
                    n_init=n_inits[bo_type],n_iter=n_iters[bo_type],
                    n_step=21,bo_type=bo_type,
                    problem_name='mpb',
                    experiment_name=experiment_name,
                    approximation=None
                    )
    '''
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--n-var', type=int, default=2,
                        help='dimension of input space')
    parser.add_argument('--n-obj', type=int, default=2,
                        help='dimension of output space')
    parser.add_argument('--n-const', type=int, default=0,
                        help='number of constraints')
    parser.add_argument('--problem-name', type=str, default='ZDT1',
                        help='Test problem name')
    parser.add_argument('--n-init', type=int, default=None,
                        help='number of initial sample')
    parser.add_argument('--n-iter', type=int, default=None,
                        help='number of EFs')
    parser.add_argument('--batch', type=int, default=10,
                        help='size of batch')
    parser.add_argument('--selection-type', type=str, default='DHVI',
                        help='selection type')
    parser.add_argument('--algorithm',type=str,default='MGD')
    parser.add_argument('--algorithm-name', type=str, default='DimMOEAD',
                        help='output folder name')

    parser.add_argument('--ea-type', type=str, default='moead',
                        help='EA used in Dim')


    args, _ = parser.parse_known_args(None)
    # Setup default args
    args.n_init = (args.n_var * 11 - 1) if args.n_init is None else args.n_init
    args.n_iter = 10*(args.n_var * 11 - 1) if args.n_iter is None else args.n_iter
    return args, parser

