import logging
from time import perf_counter

import numpy as np
from spn.algorithms.Inference import likelihood
from spn.structure.Base import Product

from rspn.code_generation.convert_conditions import convert_range
from rspn.structure.base import Sum, Sample

###
from rspn.structure.leaves.cltree.CLTree import CLTree
from rspn.structure.leaves.cltree.Inference import cltree_log_likelihood
from rspn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear

logger = logging.getLogger(__name__)


def expectation(spn, feature_scope, inverted_features, ranges, node_expectation=None, node_likelihoods=None,
                use_generated_code=False, spn_id=None, meta_types=None, gen_code_stats=None,
                outer_join_table=None):
    """Compute the Expectation:
        E[1_{conditions} * X_feature_scope]
        First factor is one if condition is fulfilled. For the second factor the variables in feature scope are
        multiplied. If inverted_features[i] is True, variable is taken to denominator.
        The conditional expectation would be E[1_{conditions} * X_feature_scope]/P(conditions)
    """

    # evidence_scope = set([i for i, r in enumerate(ranges) if not np.isnan(r)])
    evidence_scope = set([i for i, r in enumerate(ranges[0]) if r is not None])
    evidence = ranges

    assert not (len(evidence_scope) > 0 and evidence is None)

    relevant_scope = set()
    relevant_scope.update(evidence_scope)
    relevant_scope.update(feature_scope)
    if len(relevant_scope) == 0:
        return np.ones((ranges.shape[0], 1))

    if ranges.shape[0] == 1:

        applicable = True
        if use_generated_code:
            boolean_relevant_scope = [i in relevant_scope for i in range(len(meta_types))]
            boolean_feature_scope = [i in feature_scope for i in range(len(meta_types))]
            applicable, parameters = convert_range(boolean_relevant_scope, boolean_feature_scope, meta_types, ranges[0],
                                                   inverted_features)

        # generated C++ code
        if use_generated_code and applicable:
            time_start = perf_counter()
            import optimized_inference

            spn_func = getattr(optimized_inference, f'spn{spn_id}')
            result = np.array([[spn_func(*parameters)]])

            time_end = perf_counter()

            if gen_code_stats is not None:
                gen_code_stats.calls += 1
                gen_code_stats.total_time += (time_end - time_start)

            # logger.debug(f"\t\tGenerated Code Latency: {(time_end - time_start) * 1000:.3f}ms")
            return result

        # lightweight non-batch version
        else:
            return np.array(
                [[expectation_recursive(spn, feature_scope, inverted_features, relevant_scope, evidence,
                                        node_expectation, node_likelihoods,outer_join_table=outer_join_table)]])
    # full batch version
    return expectation_recursive_batch(spn, feature_scope, inverted_features, relevant_scope, evidence,
                                       node_expectation, node_likelihoods)


def expectation_recursive_batch(node, feature_scope, inverted_features, relevant_scope, evidence, node_expectation,
                                node_likelihoods):
    if isinstance(node, Product):

        llchildren = np.concatenate(
            [expectation_recursive_batch(child, feature_scope, inverted_features, relevant_scope, evidence,
                                         node_expectation, node_likelihoods)
             for child in node.children if
             len(relevant_scope.intersection(child.scope)) > 0], axis=1)
        return np.nanprod(llchildren, axis=1).reshape(-1, 1)

    elif isinstance(node, Sum):
        if len(relevant_scope.intersection(node.scope)) == 0:
            return np.full((evidence.shape[0], 1), np.nan)

        llchildren = np.concatenate(
            [expectation_recursive_batch(child, feature_scope, inverted_features, relevant_scope, evidence,
                                         node_expectation, node_likelihoods)
             for child in node.children], axis=1)

        relevant_children_idx = np.where(np.isnan(llchildren[0]) == False)[0]
        if len(relevant_children_idx) == 0:
            return np.array([np.nan])

        weights_normalizer = sum(node.weights[j] for j in relevant_children_idx)
        b = np.array(node.weights)[relevant_children_idx] / weights_normalizer

        return np.dot(llchildren[:, relevant_children_idx], b).reshape(-1, 1)

    # elif isinstance(node, CLTree):
    #     from spn.structure.leaves.cltree.MPE import cltree_bottom_up_log_ll
    #
    #     return cltree_bottom_up_log_ll(node, node.data)
    # #     pass

    elif isinstance(node, PiecewiseLinear):
        from spn.algorithms.stats.Expectations import Expectation

        # return Expectation(node, feature_scope)
        from spn.algorithms.stats.Expectations import Expectation

        # return Expectation(node, feature_scope)
        if node.scope[0] in feature_scope:
            t_node = type(node)
            if t_node in node_expectation:

                feature_idx = feature_scope.index(node.scope[0])
                inverted = inverted_features[feature_idx]

                return Expectation(node, feature_scope= node.scope, evidence=evidence)
            else:
                return Expectation(node, feature_scope= node.scope, evidence=evidence)
                # raise Exception('Node type unknown: ' + str(t_node))

        return 1

    elif isinstance(node, Sample):
        if len(set(node.scope).intersection(feature_scope)) > 0:
            from rspn.algorithms.ranges import NominalRange, NumericRange
            import pandas as pd

            required_data = []
            node_data = pd.DataFrame(node.data)

            i = 0
            for ranges in evidence:
                if ranges is not None:
                    if isinstance(ranges, NominalRange) and i in node.scope:
                        required_data = node_data[node_data[:, i] in ranges]
                    elif isinstance(ranges, NumericRange) and i in node.scope:
                        for j, single_range in ranges.get_ranges():
                            required_data = required_data[required_data[:, i].isin(single_range)]
                i += 1

            return len(required_data)
        else:
            return 1

    else:
        if node.scope[0] in feature_scope:
            t_node = type(node)
            if t_node in node_expectation:
                exps = np.zeros((evidence.shape[0], 1))

                feature_idx = feature_scope.index(node.scope[0])
                inverted = inverted_features[feature_idx]

                exps[:] = node_expectation[t_node](node, evidence, inverted=inverted)
                return exps
            else:
                raise Exception('Node type unknown: ' + str(t_node))

        return likelihood(node, evidence, node_likelihood=node_likelihoods)


def nanproduct(product, factor):
    if np.isnan(product):
        if not np.isnan(factor):
            return factor
        else:
            return np.nan
    else:
        if np.isnan(factor):
            return product
        else:
            return product * factor


def expectation_recursive(node, feature_scope, inverted_features, relevant_scope, evidence, node_expectation,
                          node_likelihoods, parent=None, outer_join_table=None):
    # print(type(node))
    if isinstance(node, Product):

        # if len(relevant_scope.intersection(node.scope)) == 0:
        #     return np.nan

        product = np.nan
        for child in node.children:
            if len(relevant_scope.intersection(child.scope)) > 0:
                # print(type(child))
                # from spn.structure.Base import Sum
                # print(type(Sum))
                # print(isinstance(child, Sum))
                factor = expectation_recursive(child, feature_scope, inverted_features, relevant_scope, evidence,
                                               node_expectation, node_likelihoods, node, outer_join_table=outer_join_table)
                product = nanproduct(product, factor)
        return product

    elif isinstance(node, Sum):
        if len(relevant_scope.intersection(node.scope)) == 0:
            return np.nan

        llchildren = [expectation_recursive(child, feature_scope, inverted_features, relevant_scope, evidence,
                                            node_expectation, node_likelihoods, node, outer_join_table=outer_join_table)
                      for child in node.children]

        relevant_children_idx = np.where(np.isnan(llchildren) == False)[0]

        if len(relevant_children_idx) == 0:
            return np.nan

        weights_normalizer = sum(node.weights[j] for j in relevant_children_idx)
        weighted_sum = sum(node.weights[j] * llchildren[j] for j in relevant_children_idx)

        return weighted_sum / weights_normalizer

    # elif isinstance(node, CLTree):
    #     from rspn.structure.leaves.cltree.MPE import cltree_bottom_up_log_ll
    #
    #     return cltree_bottom_up_log_ll(node, node.data)

    elif isinstance(node, PiecewiseLinear):
        from spn.algorithms.stats.Expectations import Expectation

        # return Expectation(node, feature_scope)
        if node.scope[0] in feature_scope:
            t_node = type(node)
            if t_node in node_expectation:

                feature_idx = feature_scope.index(node.scope[0])
                inverted = inverted_features[feature_idx]

                return Expectation(node, feature_scope= node.scope, evidence=evidence)
            else:
                return Expectation(node, feature_scope= node.scope, evidence=evidence)
                # raise Exception('Node type unknown: ' + str(t_node))

        return 1

    elif isinstance(node, Sample):
        if len(set(node.scope).intersection(relevant_scope)) > 0:
            from rspn.algorithms.ranges import NominalRange, NumericRange
            import pandas as pd

            # required_data = []
            col = []
            for i in node.scope:
                col.append(str(i))

            required_data = pd.DataFrame(node.data, columns=col)

            i = 0
            # for ranges in evidence[0]:
            #     if ranges is not None:
            #         if isinstance(ranges, NominalRange) and i in node.scope:
            #             # b = len(required_data)
            #             required_data = required_data[required_data[str(i)].isin(ranges.get_ranges())]
            #             # c = len(required_data)
            #             #
            #             # if b != c:
            #             #     raise "success"
            #         elif isinstance(ranges, NumericRange) and i in node.scope:
            #             for single_range in ranges.get_ranges():
            #                 required_data = required_data[(single_range[0] <= required_data[str(i)])
            #                                               & (required_data[str(i)] <= single_range[1])]
            #
            #     i += 1
            # return np.nan
            # return  len(required_data) / len(node.data) #result should be Sum( 1 / (F(Q,J)) * Lc * product( Nt ...) ... )
            # return cal_exp(node, required_data, evidence, relevant_scope, feature_scope, inverted_features) / len(node.data)
            return np.nan

        else:
            return np.nan

    else:
        if node.scope[0] in feature_scope:
            t_node = type(node)
            if t_node in node_expectation:

                feature_idx = feature_scope.index(node.scope[0])
                inverted = inverted_features[feature_idx]

                # tx = node_expectation[t_node](node, evidence, inverted=inverted).item()
                return node_expectation[t_node](node, evidence, inverted=inverted).item()
                # return 1
            else:
                raise Exception('Node type unknown: ' + str(t_node))

        # return node_likelihoods[type(node)](node, evidence, outer_join_table).item()
        return node_likelihoods[type(node)](node, evidence).item()

def lc(node, col, evidence, relevant_scope, feature_scope):
    from rspn.algorithms.ranges import NominalRange, NumericRange
    import pandas as pd

    scopes = list(set(relevant_scope) - set(feature_scope))
    if len(scopes) < 0 :
        raise "Error! relevant_scope should be larger than feature_scope"

    i = 0
    for ranges in evidence[0]:
        if ranges is not None and i in node.scope:
            idx = node.scope.index(i)
            if isinstance(ranges, NominalRange):
                if col[:, idx].item() in ranges.get_ranges():
                    continue
                else:
                    return 0.0
            elif isinstance(ranges, NumericRange):
                a = col[:, idx].item()
                b = ranges.get_ranges()
                if ranges.get_ranges()[0][0] <= col[:, idx].item() <= ranges.get_ranges()[0][1]:
                    continue
                else:
                    return 0.0
        i += 1


    return 1.0


def cal_exp(node, data, evidence, relevant_scope, feature_scope, inverted_features):
    import pandas as pd

    exp = 0.0

    for idx, col in data.iterrows():
        col = col.values.reshape(1, col.values.shape[0])
        # col = pd.DataFrame(col, columns=node.scope)
        per_col_exp = 1
        if len(set(node.scope).intersection(feature_scope)) > 0:
            for scope in feature_scope:
                if scope in node.scope:
                    feature_idx = feature_scope.index(scope)
                    inverted = inverted_features[feature_idx]

                    col_idx = node.scope.index(scope)
                    if not inverted:
                        per_col_exp *= col[:, col_idx]
                    else:
                        per_col_exp *= 1 / col[:, col_idx]

        per_col_exp *= lc(node, col, evidence, relevant_scope, feature_scope)

        exp += per_col_exp

    return exp


