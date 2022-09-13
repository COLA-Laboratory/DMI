import numpy as np
from copy import deepcopy
from scipy.spatial import Delaunay
from utils.common import tchebycheff
from utils.lhs import lhs
from pygco import cut_from_graph


class WeightAssociatedSamples:
    def __init__(self,weight,n_var,n_obj):
        self.weight = weight
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_weight = weight.shape[0]
        self.max_associated = 10  # maximum solutions associated on a single weight
        self.x = [[] for _ in range(self.n_weight)]
        self.y = [[] for _ in range(self.n_weight)]
        self.dist = [[] for _ in range(self.n_weight)]
        self.idx = [[] for _ in range(self.n_weight)]
        self.n_sample = 0
        self.ideal_point = None
        self.eps = 1e-2
        self.nadir_point = None
        self.vecs = self.weight / np.tile(np.linalg.norm(self.weight, axis=1)[:, None], (1, self.weight.shape[1]))

    def insert(self, X, Y, idx, duplicated=False):
        X, Y = np.array(X), np.array(Y)
        if self.ideal_point is not None:
            self.set_ideal_point(np.min(Y, axis=0))
        if self.nadir_point is not None:
            self.set_nadir_point(np.max(Y, axis=0))

        if self.nadir_point is None and self.ideal_point is None:
            norm_Y = Y
        elif self.nadir_point is not None and self.ideal_point is not None:
            norm_Y = (Y - self.ideal_point) / (self.nadir_point - self.ideal_point)
        elif self.nadir_point is None and self.ideal_point is not None:
            norm_Y = Y - self.ideal_point
        else:
            print('Warning: wrong ideal/nardir setting in association')
            norm_Y = Y
        dist = np.linalg.norm(norm_Y, axis=1)

        weight_id, tcheb = self._associated_sample_to_weight(norm_Y)
        # TODO: should use tcheb instead of dist

        for x, y, w_id, d, i in zip(X, Y, weight_id, dist, idx):
            self.x[w_id].append(x)
            self.y[w_id].append(y)
            self.dist[w_id].append(d)
            self.idx[w_id].append(i)
        self.n_sample += X.shape[0]

        for w_id in np.unique(weight_id):
            self._sort_single_weight(w_id)

        if duplicated:
            # we duplicated samples making every weight have at least one solutions
            empty_weight_id = np.delete(np.arange(self.n_weight),np.unique(weight_id))
            sample_id, tcheb = self._associated_weight_to_sample(X)
            for w_id in empty_weight_id:
                self.x[w_id].append(X[sample_id[w_id]])
                self.y[w_id].append(Y[sample_id[w_id]])
                self.dist[w_id].append(dist[sample_id[w_id]])
                self.idx[w_id].append(idx[sample_id[w_id]])

    def get(self,n_sample,idx = None):
        np.random.seed(0)

        selected_sample = []
        level = 0
        weight_id = []
        for w_id in range(self.n_weight):
            if len(self.dist[w_id]) > level:
                weight_id.append(w_id)

        if n_sample <= len(weight_id):
            weight_id = np.random.choice(weight_id, n_sample, replace=False)
            for w_id in weight_id:
                if idx is None or self.idx[w_id][level] == idx:
                    selected_sample.append(self.x[w_id][0])
        else:
            while len(selected_sample) < n_sample:
                weight_id = []
                for w_id in range(self.n_weight):
                    if len(self.dist[w_id]) > level:
                        weight_id.append(w_id)
                if len(weight_id) == 0: # dont have sample anymore
                    print(f'Warning: Not enough sample in the association, adding {n_sample - len(selected_sample)} random samples')
                    addition_sample = lhs(self.n_var, n_sample - len(selected_sample))
                    selected_sample = np.vstack([selected_sample, addition_sample])
                else:
                    for w_id in weight_id:
                        if idx is None or self.idx[w_id][level] == idx:
                            selected_sample.append(self.x[w_id][level])
                level += 1
        return np.vstack(selected_sample)[:n_sample]

    def get_all(self):
        xs, ys, ds, idx = [], [], [], []
        for x, y, d, i in zip(self.x, self.y, self.dist, self.idx):
            if x:
                xs.append(x)
            if y:
                ys.append(y)
            if y:
                ds.append(d)
            if i:
                idx.append(i)

        return np.concatenate(xs), np.concatenate(ys), np.concatenate(ds), idx

    def set_ideal_point(self,new_ideal_point):
        if self.ideal_point is not None:
            if (new_ideal_point >= self.ideal_point).all() and not (new_ideal_point == self.ideal_point).any():
                return
            self.ideal_point = np.minimum(self.ideal_point, new_ideal_point) - self.eps
        else:
            self.ideal_point = new_ideal_point.squeeze()
        xs = deepcopy(self.x)
        self.x = [[] for _ in range(self.n_weight)]
        ys = deepcopy(self.y)
        self.y = [[] for _ in range(self.n_weight)]
        self.dist = [[] for _ in range(self.n_weight)]
        idx = deepcopy(self.idx)
        self.idx = [[] for _ in range(self.n_weight)]
        self.n_sample = 0
        for (x,y,i) in zip(xs,ys,idx):
            if len(x) >0:
                self.insert(x,y,i)

    def set_nadir_point(self,new_nadir_point):
        if self.nadir_point is not None:
            if (new_nadir_point >= self.ideal_point).all() and not (new_nadir_point == self.ideal_point).any():
                return
            self.nadir_point = np.maximum(self.nadir_point, new_nadir_point) + self.eps
        else:
            self.nadir_point = new_nadir_point.squeeze()
        xs = deepcopy(self.x)
        self.x = [[] for _ in range(self.n_weight)]
        ys = deepcopy(self.y)
        self.y = [[] for _ in range(self.n_weight)]
        self.dist = [[] for _ in range(self.n_weight)]
        idx = deepcopy(self.idx)
        self.idx = [[] for _ in range(self.n_weight)]
        self.n_sample = 0
        for (x,y,i) in zip(xs,ys,idx):
            if len(x) >0:
                self.insert(x,y,i)

    def sparse_approximation(self):
        '''
        Use a few manifolds to sparsely approximate the pareto front by graph-cut, see section 6.4.
        Output:
            labels: the optimized labels (manifold index) for each non-empty cell (the cells also contain the corresponding labeled sample), shape = (n_label,)
            approx_x: the labeled design samples, shape = (n_label, n_var)
            approx_y: the labeled performance values, shape = (n_label, n_obj)
        '''
        # update patch ids, remove non-existing ids previously removed from buffer
        mapping = {}
        patch_id_count = 0
        for w_id in range(self.n_weight):
            if len(self.idx[w_id]) == 0:
                continue

            curr_patches = self.idx[w_id]
            for i in range(len(curr_patches)):
                if curr_patches[i] not in mapping:
                    mapping[curr_patches[i]] = patch_id_count
                    patch_id_count += 1
                self.idx[w_id][i] = mapping[curr_patches[i]]

        valid_weight = np.where([self.idx[cell_id] != [] for cell_id in range(self.n_weight)])[0]  # non-empty cells
        n_node = len(valid_weight)
        n_label = patch_id_count
        unary_cost = 10 * np.ones((n_node, n_label))
        pairwise_cost = -10 * np.eye(n_label)

        for i, idx in enumerate(valid_weight):
            patches, distances = np.array(self.idx[idx]), np.array(self.dist[idx])
            min_dist = np.min(distances)
            unary_cost[i, patches] = np.minimum((distances - min_dist) / 0.2, 10)

        # get edge information (graph structure)
        edges = self._get_graph_edges(valid_weight)

        # NOTE: pygco only supports int32 as input, due to potential numerical error
        edges, unary_cost, pairwise_cost = edges.astype(np.int32), unary_cost.astype(np.int32), pairwise_cost.astype(np.int32)

        # do graph-cut, optimize labels for each valid cell
        labels_opt = cut_from_graph(edges, unary_cost, pairwise_cost, 10)

        # find corresponding design and performance values of optimized labels for each valid cell
        approx_xs, approx_ys, approx_dist = [], [], []
        approx_id = []  # for a certain cell, there could be no sample belongs to that label, probably due to the randomness of sampling or improper energy definition
        for idx, label in zip(valid_weight, labels_opt):
            for x,y,d,i in zip(self.x[idx],  self.y[idx],self.dist[idx],self.idx[idx]):
                # since each buffer element array is sorted based on distance to origin
                if i == label:
                    approx_xs.append(x)
                    approx_ys.append(y)
                    approx_dist.append(d)
                    approx_id.append(label)
                    break
                else:
                    approx_xs.append(self.x[idx][0])
                    approx_ys.append(self.y[idx][0])
                    approx_id.append(label)
        approx_xs, approx_ys = np.array(approx_xs), np.array(approx_ys)
        return  approx_xs, approx_ys, approx_dist, approx_id

    def _sort_single_weight(self,w_id):
        if len(self.x[w_id]) == 0:
            return
        idx = np.argsort(self.dist[w_id])
        self.n_sample -= max(len(idx) - self.max_associated, 0)
        idx = idx[:self.max_associated]

        self.x[w_id] = list(np.array(self.x[w_id])[idx])
        self.y[w_id] = list(np.array(self.y[w_id])[idx])
        self.dist[w_id] = list(np.array(self.dist[w_id])[idx])
        self.idx[w_id] = list(np.array(self.idx[w_id])[idx])

    def _get_graph_edges(self, valid_cells):
        # get edges by connecting neighbor cells
        if self.n_obj == 2:
            edges = np.array([[i, i + 1] for i in range(len(valid_cells) - 1)])
        elif self.n_obj == 3:
            if len(valid_cells) == 1:
                raise Exception('only 1 non-empty cell in buffer, cannot do graph cut')
            elif len(valid_cells) == 2:
                return np.array([[0, 1]])
            elif len(valid_cells) == 3:
                return np.array([[0, 1], [0, 2], [1, 2]])

            # triangulate endpoints of cell vectors to form a mesh, then get edges from this mesh
            vertices = self.vecs[valid_cells]

            # check if vertices fall on a single line
            check_equal = (vertices == vertices[0]).all(axis=0)
            if check_equal.any():
                indices = np.argsort(vertices[:, np.where(np.logical_not(check_equal))[0][0]])
                edges = np.array([indices[:-1], indices[1:]]).T
                edges = np.ascontiguousarray(edges)
                return edges

            tri = Delaunay(vertices)
            ind, all_neighbors = tri.vertex_neighbor_vertices
            edges = []
            for i in range(len(vertices)):
                neighbors = all_neighbors[ind[i]:ind[i + 1]]
                for j in neighbors:
                    edges.append(np.sort([i, j]))
            edges = np.unique(edges, axis=0)

        else:
            print("Error not implement in _get_graph_edges")
            edges = None
        return edges

    def _associated_sample_to_weight(self,X):
        n_sample = len(X)
        if self.n_obj == 2:
            return np.minimum(
                (np.arccos(X[:, 0] / np.linalg.norm(X, axis=1)) / (0.5 * np.pi)  * self.n_weight).astype(np.int),
                self.n_weight - 1), np.zeros((n_sample,))
        else:
            associated_id = np.ones((n_sample,),dtype=np.int) * -1
            associated_dist = np.ones((n_sample,)) * -1
            for i in range(n_sample):
                associated_id[i] = np.argmin(tchebycheff(X[i],self.weight))
                associated_dist[i] = np.min(tchebycheff(X[i],self.weight))
            return associated_id,associated_dist

    def _associated_weight_to_sample(self, X, weights=None):
        if weights is None:
            weights = self.weight
        weights = np.atleast_2d(weights)
        n_sample = len(X)
        n_weight = len(weights)
        associated_id = np.ones((n_weight,),dtype=np.int) * -1
        associated_dist = np.ones((n_weight,)) * -1
        for i in range(n_weight):
            associated_id[i] = np.argmin(tchebycheff(X,weights[i]))
            associated_dist[i] = np.min(tchebycheff(X,weights[i]))
        return associated_id,associated_dist

