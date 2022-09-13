import datetime
import time
import os
import numpy as np

from problems.zdt import ZDT1,ZDT2,ZDT3,ZDT4,ZDT31,ZDT32,ZDT33,ZDT34,ZDT6
from problems.dtlz import DTLZ1,DTLZ2,DTLZ3,DTLZ4,DTLZ7,DTLZ71,DTLZ72, DTLZ73
from problems.wfg import WFG1,WFG2,WFG3,WFG4,WFG5,WFG6,WFG21,WFG22,WFG23,WFG24,WFG42,WFG43,WFG44,WFG45,WFG46,WFG47,WFG48
from problems.idtlz import IDTLZ1,IDTLZ2,IDTLZ3,IDTLZ4
from problems.mdtlz import MDTLZ1,MDTLZ2,MDTLZ3,MDTLZ4

from utils.plot import plot2D,plot3D


class MultiObjectiveOptimization:
    def __init__(self, args):
        self.n_var = args.n_var
        self.n_obj = args.n_obj
        self.n_init = args.n_init
        self.n_iter = args.n_iter
        self.problem_name = args.problem_name
        self.problem = self._problem_chooser(args.problem_name, args)

        self.seed = args.seed
        self.algorithm_name = self.__class__.__name__ if (
                    'algorithm_name' not in args or args.algorithm_name is None) else args.algorithm_name
        if not hasattr(args, 'output') or args.output is None:
            self.output = {'timer': True, 'terminal': True, 'plot':True}
        else:
            self.output = args.output
        np.random.seed(self.seed)
        self.time_recorder = None
        self.folder_name = {
            'fig': f'output/figs/{self.problem_name}/x{self.n_var}y{self.n_obj}/' +
                   '/' + self.algorithm_name + '/' + f'{self.seed}' + '/',
            'data': f'output/data/{self.problem_name}/x{self.n_var}y{self.n_obj}/' +
                    '/' + self.algorithm_name + '/' + f'{self.seed}' + '/',
            'debug': f'output/debug/{self.problem_name}/x{self.n_var}y{self.n_obj}/' +
                     '/' + self.algorithm_name + '/' + f'{self.seed}' + '/',
            'log': f'output/log/{self.problem_name}/x{self.n_var}y{self.n_obj}/' +
                   '/' + self.algorithm_name + '/' + f'{self.seed}' + '/'}
        os.makedirs(self.folder_name['fig'] + 'internal/', exist_ok=True)
        os.makedirs(self.folder_name['data'], exist_ok=True)
        os.makedirs(self.folder_name['debug'], exist_ok=True)
        os.makedirs(self.folder_name['log'], exist_ok=True)

    def _after_init(self, optimizer):
        # print('In _after_init')
        if 'timer' in self.output and self.output['timer']:
            self.time_recorder = {'begin': time.time(), 'fit_all': 0.0, 'propose_all': 0.0,
                                  'fit_last': 0.0, 'propose_last': 0.0}
        return

    def _before_fit(self, optimizer):
        # print('In _before_fit')

        if 'timer' in self.output and self.output['timer']:
            self.time_recorder['fit_last'] = time.time()

        return

    def _after_fit(self, optimizer):
        # print('In _after_fit')
        if 'timer' in self.output and self.output['timer']:
            self.time_recorder['fit_all'] += (time.time() - self.time_recorder['fit_last'])
            self.time_recorder['fit_last'] = time.time()

        return

    def _before_propose(self, optimizer):
        # print('In _before_proposed')
        if 'timer' in self.output and self.output['timer']:
            self.time_recorder['propose_last'] = time.time()

        return

    def _user_define_propose(self, optimizer):
        optimizer.suggested_sample = optimizer._compute_next_evaluations()

    def _after_propose(self, optimizer):
        if 'timer' in self.output and self.output['timer']:
            self.time_recorder['propose_all'] += (time.time() - self.time_recorder['propose_last'])
            self.time_recorder['propose_last'] = time.time()
        return

    def _end_iteration(self, optimizer):
        if 'terminal' in self.output and self.output['terminal']:
            if 'timer' in self.output and self.output['timer']:
                t0 = str(datetime.timedelta(seconds=int(time.time() - self.time_recorder['begin'])))
                t1 = str(datetime.timedelta(seconds=int(self.time_recorder['fit_all'])))
                t2 = str(datetime.timedelta(seconds=int(self.time_recorder['propose_all'])))
                print(f'Info:: Timer[{t0}/{t1}/{t2}(all/fit/propose)]')

        if 'plot' in self.output and self.output['plot']:
            ax = None
            if self.n_obj == 2:
                ax = plot2D(self.problem.get_pareto_front()[:, 0], self.problem.get_pareto_front()[:, 1],
                            ls='-', marker='', label='pf', c='black', ax=ax)
                ax = plot2D(optimizer.Y[:self.n_init, 0], optimizer.Y[:self.n_init, 1],
                            ls='', marker='x', label='inits', c='gray', ax=ax)
                ax = plot2D(optimizer.Y[self.n_init:-optimizer.next_size, 0], optimizer.Y[self.n_init:-optimizer.next_size, 1],
                            ls='', marker='o', label='Samples', c='gray', ax=ax)
                plot2D(optimizer.Y[-optimizer.next_size:, 0], optimizer.Y[-optimizer.next_size:, 1], ls='', marker='o', label='next', c='red', ax=ax,
                       file=self.folder_name['fig'] + f'/internal/EF{optimizer.Y.shape[0]}.pdf', show_legend=True,
                       )
            if self.n_obj == 3:
                ax = plot3D(self.problem.get_pareto_front()[:, 0],
                            self.problem.get_pareto_front()[:, 1],
                            self.problem.get_pareto_front()[:, 2],
                            ls='', marker=',', label='pf', c='black', ax=ax)
                ax = plot3D(optimizer.Y[:self.n_init, 0], optimizer.Y[:self.n_init, 1], optimizer.Y[:self.n_init, 2],
                            ls='', marker='x', label='inits', c='gray', ax=ax)
                ax = plot3D(optimizer.Y[self.n_init:-optimizer.next_size, 0],
                            optimizer.Y[self.n_init:-optimizer.next_size, 1],
                            optimizer.Y[self.n_init:-optimizer.next_size, 2],
                            ls='', marker='o', label='Samples', c='gray', ax=ax)
                plot3D(optimizer.Y[-optimizer.next_size:, 0],
                       optimizer.Y[-optimizer.next_size:, 1],
                       optimizer.Y[-optimizer.next_size:, 2],
                       ls='', marker='o',
                       label='next', c='red', ax=ax,
                       file=self.folder_name['fig'] + f'/internal/EF{optimizer.Y.shape[0]}.pdf', show_legend=True,
                       )

        return

    def _after_finish(self, optimizer):

        np.savetxt(self.folder_name['data'] + 'X.txt',optimizer.X)
        np.savetxt(self.folder_name['data'] + 'Y.txt', optimizer.Y)

        if 'plot' in self.output and self.output['plot']:
            ax = None
            if self.n_obj == 2:
                ax = plot2D(self.problem.get_pareto_front()[:, 0], self.problem.get_pareto_front()[:, 1],
                            ls='-', marker='', label='pf', c='black', ax=ax)
                plot2D(optimizer.fx_opt[:, 0], optimizer.fx_opt[:, 1], ls='', marker='o', label='Samples', c='blue',
                       ax=ax, file=self.folder_name['fig'] + 'final.pdf', show_legend=True,
                       bounds=np.vstack((self.problem.get_ideal_point(),self.problem.get_nadir_point())).T)
            if self.n_obj == 3:
                ax = plot3D(self.problem.get_pareto_front()[:, 0], self.problem.get_pareto_front()[:, 1],
                            self.problem.get_pareto_front()[:, 2],
                            ls='', marker=',', label='pf', c='black', ax=ax)
                plot3D(optimizer.fx_opt[:, 0], optimizer.fx_opt[:, 1], optimizer.fx_opt[:, 2],
                       ls='', marker='o', label='Samples', c='blue',
                       ax=ax, file=self.folder_name['fig'] + 'final.pdf', show_legend=True,
                       bounds=np.vstack((self.problem.get_ideal_point(),self.problem.get_nadir_point())).T)
        return optimizer.callback_parameter

    @staticmethod
    def _problem_chooser(problem_name, args):
        return eval(problem_name)(args)
