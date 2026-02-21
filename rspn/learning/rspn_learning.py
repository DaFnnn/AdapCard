import logging

import numpy as np
from sklearn.cluster import KMeans
###
from rspn.learning.splitting.Base import preproc, split_data_by_clusters
from rspn.learning.splitting.RDC import getIndependentRDCGroups_py, getIndependentRDCGroups_py_test
from spn.structure.StatisticalTypes import MetaType

from rspn.structure.leave import IdentityNumericLeaf, Categorical

logger = logging.getLogger(__name__)
###
# MAX_UNIQUE_LEAF_VALUES = 10000
MAX_UNIQUE_LEAF_VALUES = 100000


def learn_mspn(
        data,
        ds_context,
        cols="rdc",
        rows="kmeans",
        min_instances_slice=200,
        threshold=0.15,
        rdc_threshold_high=0.3,
        max_sampling_threshold_cols=10000,
        max_sampling_threshold_rows=100000,
        ohe=False,
        leaves=None,
        memory=None,
        rand_gen=None,
        cpus=-1,
        table_condition=None,
        attr_condition=None
):
    """
    Adapts normal learn_mspn to use custom identity leafs and use sampling for structure learning.
    :param max_sampling_threshold_rows:
    :param max_sampling_threshold_cols:
    :param data:
    :param ds_context:
    :param cols:
    :param rows:
    :param min_instances_slice:
    :param threshold:
    :param ohe:
    :param leaves:
    :param memory:
    :param rand_gen:
    :param cpus:
    :return:
    """
    if leaves is None:
        leaves = create_custom_leaf
    if rand_gen is None:
        ### seed
        rand_gen = np.random.RandomState(11)

    from rspn.learning.structure_learning import get_next_operation, learn_structure

    def l_mspn(data, ds_context, cols, rows, min_instances_slice, threshold, ohe, rdc_threshold_high):
        split_cols, split_rows = get_splitting_functions(max_sampling_threshold_rows, max_sampling_threshold_cols, cols,
                                                         rows, ohe, threshold, rand_gen, cpus, rdc_threshold_high)

        nextop = get_next_operation(min_instances_slice)

        node = learn_structure(data, ds_context, split_rows, split_cols, leaves, next_operation=nextop,
                               table_condition=table_condition, attr_condition=attr_condition)

        return node

    if memory:
        l_mspn = memory.cache(l_mspn)

    spn = l_mspn(data, ds_context, cols, rows, min_instances_slice, threshold, ohe, rdc_threshold_high)
    return spn


def create_custom_leaf(data, ds_context, scope):
    """
    Adapted leafs for cardinality SPN. Either categorical or identityNumeric leafs.
    """

    idx = scope[0]
    meta_type = ds_context.meta_types[idx]

    if meta_type == MetaType.REAL:
        assert len(scope) == 1, "scope for more than one variable?"

        unique_vals, counts = np.unique(data[:, 0], return_counts=True)

        if hasattr(ds_context, 'no_compression_scopes') and idx not in ds_context.no_compression_scopes and \
                len(unique_vals) > MAX_UNIQUE_LEAF_VALUES:
            # if there are too many unique values build identity leaf with histogram representatives
            hist, bin_edges = np.histogram(data[:, 0], bins=MAX_UNIQUE_LEAF_VALUES, density=False)
            logger.debug(f"\t\tDue to histograms leaf size was reduced "
                         f"by {(1 - float(MAX_UNIQUE_LEAF_VALUES) / len(unique_vals)) * 100:.2f}%")
            unique_vals = bin_edges[:-1]
            probs = hist / data.shape[0]
            lidx = len(probs) - 1

            assert len(probs) == len(unique_vals)

        else:
            probs = np.array(counts, np.float64) / len(data[:, 0])
            lidx = len(probs) - 1

        null_value = ds_context.null_values[idx]
        ###
        median = np.median(data)
        leaf = IdentityNumericLeaf(unique_vals, probs, null_value, scope, cardinality=data.shape[0], median=median)

        return leaf

    elif meta_type == MetaType.DISCRETE:

        unique, counts = np.unique(data[:, 0], return_counts=True)
        # ###
        # remove blank
        # temp = unique.tolist()
        # for i in range(len(unique)):
        #     if isinstance(temp[i], str):
        #         temp[i] = temp[i].strip()
        # unique = np.array(temp)

        # +1 because of potential 0 value that might not occur
        sorted_counts = np.zeros(len(ds_context.domains[idx]) + 1, dtype=np.float64)
        for i, x in enumerate(unique):
            sorted_counts[int(x)] = counts[i]
        p = sorted_counts / data.shape[0]
        null_value = ds_context.null_values[idx]
        ###
        median = np.median(data)
        node = Categorical(p, null_value, scope, cardinality=data.shape[0], median=median)

        return node


###
# mark
def get_splitting_functions(max_sampling_threshold_rows, max_sampling_threshold_cols, cols, rows, ohe, threshold,
                            rand_gen, n_jobs, rdc_threshold_high):
    from spn.algorithms.splitting.Clustering import get_split_rows_TSNE, get_split_rows_GMM
    from spn.algorithms.splitting.PoissonStabilityTest import get_split_cols_poisson_py
    from spn.algorithms.splitting.RDC import get_split_rows_RDC_py

    if isinstance(cols, str):

        if cols == "rdc":
            split_cols = get_split_cols_RDC_py(max_sampling_threshold_cols=max_sampling_threshold_cols,
                                               # threshold=threshold, rdc_threshold_high=rdc_threshold_high,
                                               rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs)
        elif cols == "poisson":
            split_cols = get_split_cols_poisson_py(threshold, n_jobs=n_jobs)
        else:
            raise AssertionError("unknown columns splitting strategy type %s" % str(cols))
    else:
        split_cols = cols

    if isinstance(rows, str):

        if rows == "rdc":
            split_rows = get_split_rows_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs)
        elif rows == "kmeans":
            split_rows = get_split_rows_KMeans(max_sampling_threshold_rows=max_sampling_threshold_rows)
            ###
            # split_rows = get_split_rows_KMeans()
        elif rows == "tsne":
            split_rows = get_split_rows_TSNE()
        elif rows == "gmm":
            split_rows = get_split_rows_GMM()
        elif rows == "condition":
            split_rows = get_split_rows_condition()
        else:
            raise AssertionError("unknown rows splitting strategy type %s" % str(rows))
    else:
        split_rows = rows
    return split_cols, split_rows


# noinspection PyPep8Naming
def get_split_rows_KMeans(max_sampling_threshold_rows=100000, n_cluster=2, pre_proc=None, ohe=False, seed=11):
    # noinspection PyPep8Naming
    def split_rows_KMeans(local_data, ds_context, scope, n_clusters=2):
        data = preproc(local_data, ds_context, pre_proc, ohe)

        if data.shape[0] > max_sampling_threshold_rows:
            ### seed
            rnd = np.random.seed(11)
            data_sample = data[np.random.randint(data.shape[0], size=max_sampling_threshold_rows), :]

            kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
            clusters = kmeans.fit(data_sample).predict(data)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
            clusters = kmeans.fit_predict(data)

        cluster_centers = kmeans.cluster_centers_
        result = split_data_by_clusters(local_data, clusters, scope, rows=True)

        return result, cluster_centers.tolist()

    return split_rows_KMeans


###
# mark
def get_split_rows_condition():
    pass
    
    return []


# noinspection PyPep8Naming
def get_split_cols_RDC_py(max_sampling_threshold_cols=10000, threshold=0.35,
                          ohe=True, k=10, s=1/6, rdc_threshold_high=0.45, non_linearity=np.sin,
                          n_jobs=-2, rand_gen=None):
                        # 0.15, 0.35, JOB-LIGHT     rdc_threshold_high=0.5,
                        # 0.15 0.3 POWER
                        # 0.25 0.4 forest
                        # 0.15 0.35 census
                        # 0.15 0.35 dmv
    from rspn.learning.splitting.RDC import split_data_by_clusters

    def split_cols_RDC_py(local_data, ds_context, scope):
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        if local_data.shape[0] > max_sampling_threshold_cols:
            ### seed
            rnd = np.random.seed(11)
            local_data_sample = local_data[np.random.randint(local_data.shape[0], size=max_sampling_threshold_cols), :]
            clusters, group_dict, dependence_dict = getIndependentRDCGroups_py_test(
            # clusters = getIndependentRDCGroups_py(
                local_data_sample,
                threshold,
                # 0.15,
                meta_types,
                domains,
                k=k,
                s=s,
                # ohe=True,
                non_linearity=non_linearity,
                n_jobs=n_jobs,
                rand_gen=rand_gen,
                # rdc_threshold_high=rdc_threshold_high
            )
            return split_data_by_clusters(local_data, clusters, scope, rows=False)

        else:
            clusters, group_dict, dependence_dict = getIndependentRDCGroups_py_test(
            # clusters = getIndependentRDCGroups_py(
                local_data,
                threshold,
                # 0.15,
                meta_types,
                domains,
                k=k,
                s=s,
                # ohe=True,
                non_linearity=non_linearity,
                n_jobs=n_jobs,
                rand_gen=rand_gen,
                # rdc_threshold_high=rdc_threshold_high
            )
            return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_RDC_py

def get_split_cols_RDC_py_test(max_sampling_threshold_cols=50000, threshold=0.15, rdc_threshold_high = 0.45,
                          ohe=True, k=10, s=1 / 6,
                          non_linearity=np.sin,
                          n_jobs=-2, rand_gen=None):
    from rspn.learning.splitting.RDC import split_data_by_clusters

    def split_cols_RDC_py_test(local_data, ds_context, scope):
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        if local_data.shape[0] > max_sampling_threshold_cols:
            ### seed
            rnd = np.random.seed(11)
            local_data_sample = local_data[np.random.randint(local_data.shape[0], size=max_sampling_threshold_cols), :]

            n_features = local_data_sample.shape[1]
            result = np.zeros(n_features)

            result[:n_features//2] = 1
            result[n_features//2:n_features] = 2

            return split_data_by_clusters(local_data, result, scope, rows=False)

        else:
            n_features = local_data.shape[1]
            result = np.zeros(n_features)

            result[:n_features//2] = 1
            result[n_features//2:n_features] = 2

            return split_data_by_clusters(local_data, result, scope, rows=False)

    return split_cols_RDC_py_test
