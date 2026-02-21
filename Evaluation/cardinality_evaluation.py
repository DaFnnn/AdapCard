import csv
import math
import logging
from time import perf_counter

import numpy as np
import pandas as pd

from DataPreparation.graph_representation import QueryType
from DataPreparation.physical_db import DBConnection, TrueCardinalityEstimator
from Structure.spn_ensemble import read_ensemble
from Evaluation.utils import parse_query, save_csv

logger = logging.getLogger(__name__)


def compute_ground_truth(query_filename, target_path, physical_db_name):
    """
    Queries database for each query and stores result rows in csv file.
    :param query_filename: where to take queries from
    :param target_path: where to store dictionary
    :param physical_db_name: name of the database
    :return:
    """

    db_connection = DBConnection(db=physical_db_name)

    # read all queries
    with open(query_filename) as f:
        queries = f.readlines()

    csv_rows = []
    for query_no, query_str in enumerate(queries):
        logger.debug(f"Computing ground truth for cardinality query {query_no}: {query_str}")
        query_str = query_str.strip()
        cardinality_true = db_connection.get_result(query_str)

        csv_rows.append({'query_no': query_no,
                         'query': query_str,
                         'cardinality_true': cardinality_true})

    save_csv(csv_rows, target_path)


class GenCodeStats:

    def __init__(self):
        self.calls = 0
        self.total_time = 0.0


def evaluate_cardinalities(ensemble_location, physical_db_name, query_filename, target_csv_path, schema,
                           rdc_spn_selection, pairwise_rdc_path, use_generated_code=False,
                           true_cardinalities_path=None,
                           max_variants=1, merge_indicator_exp=False, exploit_overlapping=False, min_sample_ratio=0):
    """
    Loads ensemble and evaluates cardinality for every query in query_filename
    :param exploit_overlapping:
    :param min_sample_ratio:
    :param max_variants:
    :param merge_indicator_exp:
    :param target_csv_path:
    :param query_filename:
    :param true_cardinalities_path:
    :param ensemble_location:
    :param physical_db_name:
    :param schema:
    :return:
    """
    if true_cardinalities_path is not None:
        df_true_card = pd.read_csv(true_cardinalities_path, header=0)
    else:
        # True cardinality via DB
        db_connection = DBConnection(db=physical_db_name)
        true_estimator = TrueCardinalityEstimator(schema, db_connection)
        df_true_card = None

    # load ensemble
    spn_ensemble = read_ensemble(ensemble_location, build_reverse_dict=True)

    # # ### draw spn
    # from spn.io.Graphics import plot_spn
    # plot_spn(spn_ensemble.spns[0].mspn, './plot.pdf')

    # ### statistics
    # from spn.algorithms.Statistics import get_structure_stats
    # print(get_structure_stats(spn_ensemble.spns[0].mspn))

    # ### likelihood
    # from spn.algorithms.Inference import log_likelihood
    # tmp_data = pd.read_csv('/home/dafn/card/deepcard/imdb-benchmark/forest/forest.csv', header=0, escapechar='\\', encoding='utf-8', quotechar='"')
    # ll = log_likelihood(spn_ensemble.spns[0].mspn, tmp_data)
    # print(ll, np.exp(ll))

    csv_rows = []
    q_errors = []

    # read all queries
    with open(query_filename) as f:
        queries = f.readlines()

    if use_generated_code:
        spn_ensemble.use_generated_code()

    latencies = []

    #
    ##
    # col = []
    # for i in range(12):
    #     col.append(str(i))
    #
    # outer_join_table = pd.read_hdf("./imdb-benchmark/outer_join_table.hdf", key='df', mode='r')
    # outer_join_table.columns = col
    def q_error_func(true, est):
        if true == 0 and est == 0:
            return 1
        elif true == 0:
            return est
        elif est == 0:
            return true
        else:
            return max(true / est, est / true)
    #

    ###
    # # sql_with_label = open('./benchmarks/cup98_10/sql/', 'w')
    #
    # card_file = open('/home/dafn/card/deepcard/benchmarks/power/sql/power_true_card.csv', 'w')
    # card_file.write('query_no,query,cardinality_true\n')

    ### for likelihood
    # sum_of_likelihood = 1
    for query_no, query_str in enumerate(queries):
        ###
        # if query_no > 500:
        #     continue

        query_str = query_str.strip()
        logger.debug(f"Predicting cardinality for query {query_no}: {query_str}")

        query = parse_query(query_str.strip(), schema)
        assert query.query_type == QueryType.CARDINALITY
        #
        # ### for log likelihood
        # cardinality_true = 1

        if df_true_card is None:
            assert true_estimator is not None
            _, cardinality_true = true_estimator.true_cardinality(query)

        else:
            cardinality_true = \
                df_true_card.loc[df_true_card['query_no'] == query_no, ['cardinality_true']].values[0][0]

        # only relevant for generated code
        gen_code_stats = GenCodeStats()

        card_start_t = perf_counter()
        _, factors, cardinality_predict, factor_values = spn_ensemble \
            .cardinality(query, rdc_spn_selection=rdc_spn_selection, pairwise_rdc_path=pairwise_rdc_path,
                         merge_indicator_exp=merge_indicator_exp, max_variants=max_variants,
                         exploit_overlapping=exploit_overlapping, return_factor_values=True,
                         gen_code_stats=gen_code_stats)
        # outer_join_table=outer_join_table)

        card_end_t = perf_counter()
        latency_ms = (card_end_t - card_start_t) * 1000

        logger.debug(f"\t\tLatency: {latency_ms:.2f}ms")
        logger.debug(f"\t\tTrue: {cardinality_true}")
        logger.debug(f"\t\tPredicted: {cardinality_predict}")

        # ### for likelihood
        # sum_of_likelihood += math.log(cardinality_predict/59614.0)
        ###
        # q_error = q_error_func(cardinality_true, round(cardinality_predict))
        q_error = q_error_func(cardinality_true, cardinality_predict)

        logger.debug(f"Q-Error was: {q_error}")
        q_errors.append(q_error)
        csv_rows.append({'query_no': query_no,
                         'query': query_str,
                         'cardinality_predict': cardinality_predict,
                         'cardinality_true': cardinality_true,
                         'q_error': q_error,
                         'latency_ms': latency_ms,
                         'generated_spn_calls': gen_code_stats.calls,
                         'latency_generated_code': gen_code_stats.total_time * 1000})
        latencies.append(latency_ms)

        ###
        # result_file.write(str(cardinality_predict) + '\n')
        # ###
        # card_file.write(str(query_no) + ',' + query_str + ',' + str(cardinality_true) + '\n')
    ###
    # card_file.close()
    # result_file.close()

    # print percentiles of published JOB-light
    q_errors = np.array(q_errors)
    q_errors.sort()
    logger.info(f"{q_errors[-10:]}")
    # https://arxiv.org/pdf/1809.00677.pdf
    # deepdb_vals = [1.25, 3.38, 4.96, 35.19, 2.71]
    # 50 70 90 100
    # deepdb_vals = [1.30, 1.60, 3.46, 40.64, 2.76]

    # 50 90 95 100
    deepdb_vals = [1.30, 111, 3.46, 5.09, 111, 40.64, 2.76]
    # mcsn_vals = [3.82, 78.4, 362, 927, 57.9]
    for i, percentile in enumerate([50, 70, 90, 95, 99, 100]):
        logger.info(f"Q-Error {percentile}%-Percentile: {np.percentile(q_errors, percentile):.3f} (vs. "
                    f"DeepDB: {deepdb_vals[i]})")
                    # f"MCSN: {mcsn_vals[i]} and DeepDB: {deepdb_vals[i]})")

    logger.info(f"Q-Mean wo inf {np.mean(q_errors[np.isfinite(q_errors)]):.3f} (vs. "
                f"DeepDB: {deepdb_vals[-1]})")
                # f"MCSN: {mcsn_vals[-1]} and DeepDB: {deepdb_vals[-1]})")
    logger.info(f"Latency avg: {np.mean(latencies):.3f}ms")

    # for best threshold
    # # 新增文件追加写入逻辑
    from datetime import datetime
    import os

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_name = os.path.basename(os.path.dirname(query_filename))

    LOG_FILE = "/home/dafn/card/deepcard/evaluation_summary_medium_correlated_parts.log"
    with open(LOG_FILE, 'a') as f:
        f.write(# f"=== Evaluation Summary ===\n"
                # f"Time: {current_time}\n"
                # f"Dataset: {dataset_name}, Table name: {schema.table_name}\n"
                # f"RDC Threshold: {}"
                f"Q-Error 50th Percentile: {np.percentile(q_errors, 50):.3f}\n"
                f"Q-Error 90th Percentile: {np.percentile(q_errors, 90):.3f}\n"
                f"Q-Error 95th Percentile: {np.percentile(q_errors, 95):.3f}\n"
                f"Q-Error 99th Percentile: {np.percentile(q_errors, 99):.3f}\n"
                f"Q-Error Mean (Finite): {np.mean(q_errors[np.isfinite(q_errors)]):.3f}\n"
                f"Latency Average: {np.mean(latencies):.3f}ms\n\n")

    # logger.info('total likelihood: ' + str(sum_of_likelihood))
    # write to csv
    save_csv(csv_rows, target_csv_path)
