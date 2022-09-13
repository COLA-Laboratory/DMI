from abc import ABC, abstractmethod
import numpy as np
from sklearn.cluster import KMeans
from pymoo.factory import get_performance_indicator
from GPyOpt.util.general import get_quantiles
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from utils.weight_associated_samples import WeightAssociatedSamples

class HVI:
    def __init__(self):
        self.ref_point = None

    def set_ref_point(self, ref_point):
        self.ref_point = ref_point

    def select(self, n_sample, xs, current_nd, model):
        xs = model.X_normalizer.undo(xs) if model[0].X_normalizer is not None else xs.copy()
        ys = []
        for m in model:
            y,_ = m.predict(xs)
            y = m.Y_normalizer.undo(y) if m.Y_normalizer is not None else y.copy()
            ys.append(y)
        ys = np.hstack(ys)

        choosen_sample = current_nd.copy()
        if self.ref_point is None:
            self.ref_point = 1.1 * np.ones((len(model)))
        hv = get_performance_indicator('hv', ref_point=self.ref_point)
        choosen_mask = np.ma.array(np.arange(len(xs)), mask=False)
        next_batch_indices = []

        for _ in range(n_sample):
            curr_hv = hv.calc(choosen_sample)
            max_hv_contrib = 0.
            max_hv_id = -1
            for i in choosen_mask.compressed():
                new_hv = hv.calc(np.vstack([choosen_sample, ys[i]]))
                hv_contrib = new_hv - curr_hv
                if hv_contrib > max_hv_contrib:
                    max_hv_contrib = hv_contrib
                    max_hv_id = i
            if max_hv_id == -1:
                max_hv_id = np.random.choice(choosen_mask.compressed())

            choosen_mask.mask[max_hv_id] = True
            choosen_sample = np.vstack([choosen_sample, ys[max_hv_id]])  # add to current pareto front
            next_batch_indices.append(max_hv_id)
        next_batch_indices = np.array(next_batch_indices)
        return xs[next_batch_indices], None


class EISelection:
    def select(self, n_sample, xs, current_nd, model):
        xs = model.X_normalizer.undo(xs) if model[0].X_normalizer is not None else xs.copy()
        f_acqus = []
        for m in model:
            y,s = m.predict(xs)
            y = m.Y_normalizer.undo(y) if m.Y_normalizer is not None else y.copy()
            fmin = m.get_fmin()
            phi, Phi, u = get_quantiles(0.01, fmin, m, s)
            ac = s * (u * Phi + phi)
            f_acqus.append(ac)
        f_acqus = np.vstack(f_acqus)
        indices = NonDominatedSorting().do(f_acqus)
        return xs[np.concatenate(indices)][:n_sample]


class RandomSelection:

    def select(self, n_sample, xs, current_nd, model):
        xs = model.X_normalizer.undo(xs) if model[0].X_normalizer is not None else xs.copy()
        random_indices = np.random.choice(len(xs), size=n_sample, replace=False)
        return xs[random_indices]


class DecomposedHVI:
    def __init__(self):
        self.nadir_point = None

    def set_nadir_point(self, ref_point):
        self.nadir_point = ref_point

    def select(self,n_sample, xs, current_nd, model, weights_id):
        xs = model[0].X_normalizer.undo(xs) if model[0].X_normalizer is not None else xs.copy()
        ys = []
        for m in model:
            # y,_ = m.predict(xs)
            y = m.predict_skgp(xs)['F']
            y = m.Y_normalizer.undo(y) if m.Y_normalizer is not None else y.copy()
            ys.append(np.expand_dims(y,-1))
        ys = np.hstack(ys)

        choosen_sample = current_nd.copy()
        if self.nadir_point is None:
            self.nadir_point = 1.1 * np.ones((len(model)))
        hv = get_performance_indicator('hv', ref_point=self.nadir_point.squeeze())
        choosen_mask = np.ma.array(np.arange(len(xs)), mask=False)
        iter_choosen_mask = np.ma.array(np.arange(len(xs)),mask=False)
        next_batch_indices = []
        weights_next = []

        for _ in range(n_sample):
            # if all families were visited, start new cycle
            if len(iter_choosen_mask.compressed()) == 0:
                iter_choosen_mask = choosen_mask.copy()
            curr_hv = hv.calc(choosen_sample)
            max_hv_contrib = 0.
            max_hv_idx = -1
            for idx in iter_choosen_mask.compressed():

                new_hv = hv.calc(np.vstack([choosen_sample, ys[idx]]))
                hv_contrib = new_hv - curr_hv
                if hv_contrib > max_hv_contrib:
                    max_hv_contrib = hv_contrib
                    max_hv_idx = idx
            if max_hv_idx == -1:
                max_hv_idx = np.random.choice(iter_choosen_mask.compressed())

            choosen_mask.mask[max_hv_idx] = True  # mask as selected
            choosen_sample = np.vstack([choosen_sample, ys[max_hv_idx]])  # add to current pareto front
            next_batch_indices.append(max_hv_idx)
            weights_next.append(weights_id[max_hv_idx])

            iter_baned_ids = np.where(weights_id == weights_id[max_hv_idx])[0]
            for fid in iter_baned_ids:
                iter_choosen_mask.mask[fid] = True

        X_next = xs[next_batch_indices].copy()
        return X_next


class DecomposedDist:
    def __init__(self):
        self.ideal_point = None
        self.nadir_point = None

    def set_nadir_point(self, ref_point):
        self.nadir_point = ref_point

    def set_ideal_point(self,ideal_point):
        self.ideal_point = ideal_point

    def select(self,n_sample, xs, current_nd_x, current_nd_y, model, weight):
        xs = model[0].X_normalizer.undo(xs) if model[0].X_normalizer is not None else xs.copy()
        ys = []
        for m in model:
            # y,_ = m.predict(xs)
            y = m.predict_skgp(xs)['F']
            ys.append(np.expand_dims(y,-1))
        ys = np.hstack(ys)

        association = WeightAssociatedSamples(weight, xs.shape[1], ys.shape[1])
        association.set_ideal_point(self.ideal_point)
        association.set_nadir_point(self.nadir_point)
        association.insert(current_nd_x,current_nd_y,np.zeros((current_nd_x.shape[0],)))

        association.insert(xs,ys,np.ones((xs.shape[0],)),duplicated=True)

        X_next = association.get(n_sample,idx=1)
        addition = 0
        while len(np.unique(X_next,axis=0)) < n_sample:
            addition += 1
            X_next = association.get(n_sample+addition, idx=1)

        return np.unique(X_next,axis=0)[:n_sample,]


