"""
Created on March 20, 2018

@author: Alejandro Molina
"""

import numpy as np
from sklearn.cluster import KMeans

from rspn.learning.splitting.Base import split_data_by_clusters
import logging

logger = logging.getLogger(__name__)
_rpy_initialized = False


# def init_rpy():
#     global _rpy_initialized
#     if _rpy_initialized:
#         return
#     _rpy_initialized = True
#
#     from rpy2 import robjects
#     from rpy2.robjects import numpy2ri
#     import os
#
#     path = os.path.dirname(__file__)
#
#     robjects.r("options(warn=-1)")
#
#     with open(path + "/rdc.R", "r") as rfile:
#         code = "".join(rfile.readlines())
#         robjects.r(code)
#
#     numpy2ri.activate()
#
#
# def get_RDC_transform(data, meta_types, ohe=False, k=10, s=1 / 6):
#     from rpy2 import robjects
#
#     init_rpy()
#
#     assert data.shape[1] == len(meta_types), "invalid parameters"
#
#     r_meta_types = [mt.name.lower() for mt in meta_types]
#
#     try:
#         df = robjects.r["as.data.frame"](data)
#         out = robjects.r["transformRDC"](df, ohe, r_meta_types, k, s)
#         out = np.asarray(out)
#     except Exception as e:
#         np.savetxt("/tmp/errordata.txt", data)
#         logger.info(e)
#         raise e
#
#     return out
#
#
# def get_RDC_adjacency_matrix(data, meta_types, ohe=False, linear=True):
#     from rpy2 import robjects
#
#     init_rpy()
#
#     assert data.shape[1] == len(meta_types), "invalid parameters"
#
#     r_meta_types = [mt.name.lower() for mt in meta_types]
#
#     try:
#         df = robjects.r["as.data.frame"](data)
#         out = robjects.r["testRDC"](df, ohe, r_meta_types, linear)
#         out = np.asarray(out)
#     except Exception as e:
#         np.savetxt("/tmp/errordata.txt", data)
#         logger.info(e)
#         raise e
#
#     return out
#
#
# def get_split_cols_RDC(threshold=0.3, ohe=True, linear=True):
#     def split_cols_RDC(local_data, ds_context, scope):
#         adjm = get_RDC_adjacency_matrix(local_data, ds_context.get_meta_types_by_scope(scope), ohe, linear)
#
#         clusters = clusters_by_adjacency_matrix(adjm, threshold, local_data.shape[1])
#
#         return split_data_by_clusters(local_data, clusters, scope, rows=False)
#
#     return split_cols_RDC
#
#
# def get_split_rows_RDC(n_clusters=2, k=10, s=1 / 6, ohe=True, seed=17):
#     def split_rows_RDC(local_data, ds_context, scope):
#         data = get_RDC_transform(local_data, ds_context.get_meta_types_by_scope(scope), ohe, k=k, s=s)
#
#         clusters = KMeans(n_clusters=n_clusters, random_state=seed, n_jobs=1).fit_predict(data)
#
#         return split_data_by_clusters(local_data, clusters, scope, rows=True)
#
#     return split_rows_RDC


############################################################################################
#
# Python version
#
import itertools

from networkx.algorithms.components.connected import connected_components
from networkx.convert_matrix import from_numpy_matrix
import scipy.stats

from sklearn.cross_decomposition import CCA
from spn.structure.StatisticalTypes import MetaType

### seed
# CCA_MAX_ITER = 100
CCA_MAX_ITER = 100


def ecdf(X):
    """
    Empirical cumulative distribution function
    for data X (one dimensional, if not it is linearized first)
    """
    # return scipy.stats.rankdata(X, method='max') / len(X)

    mv_ids = np.isnan(X)

    N = X.shape[0]
    X = X[~mv_ids]
    R = scipy.stats.rankdata(X, method="max") / len(X)
    X_r = np.zeros(N)
    X_r[~mv_ids] = R
    return X_r


def empirical_copula_transformation(data):
    ones_column = np.ones((data.shape[0], 1))
    data = np.concatenate((np.apply_along_axis(ecdf, 0, data), ones_column), axis=1)
    return data


def make_matrix(data):
    """
    Ensures data to be 2-dimensional
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]
    else:
        assert data.ndim == 2, "Data must be 2 dimensional {}".format(data.shape)

    return data


def ohe_data(data, domain):
    dataenc = np.zeros((data.shape[0], len(domain)))

    # 确保 data 和 domain 都是 NumPy 数组
    data = np.array(data)
    domain = np.array(domain)

    dataenc[data[:, None] == domain[None, :]] = 1

    #
    # this control fails when having missing data as nans
    if not np.any(np.isnan(data)):
        assert np.all((np.nansum(dataenc, axis=1) == 1)), "one hot encoding bug {} {} {}".format(
            domain, data, np.nansum(dataenc, axis=1)
        )

    return dataenc


def rdc_transformer(
    local_data,
    meta_types,
    domains,
    k=None,
    s=1.0 / 6.0,
    non_linearity=np.sin,
    return_matrix=False,
    ohe=True,
    rand_gen=None,
):
    # logger.info('rdc transformer', k, s, non_linearity)
    """
    Given a data_slice,
    return a transformation of the features data in it according to the rdc
    pipeline:
    1 - empirical copula transformation
    2 - random projection into a k-dimensional gaussian space
    3 - pointwise  non-linear transform
    """

    N, D = local_data.shape

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    #
    # precomputing transformations to reduce time complexity
    #

    #
    # FORCING ohe on all discrete features
    features = []
    for f in range(D):
        if meta_types[f] == MetaType.DISCRETE:
            features.append(ohe_data(local_data[:, f], domains[f]))
        else:
            features.append(local_data[:, f].reshape(-1, 1))
    # else:
    #     features = [data_slice.getFeatureData(f) for f in range(D)]

    #
    # NOTE: here we are setting a global k for ALL features
    # to be able to precompute gaussians
    if k is None:
        feature_shapes = [f.shape[1] if len(f.shape) > 1 else 1 for f in features]
        k = max(feature_shapes) + 1

    #
    # forcing two columness
    features = [make_matrix(f) for f in features]

    #
    # transform through the empirical copula
    features = [empirical_copula_transformation(f) for f in features]

    #
    # substituting nans with zero (the above step should have taken care of that)
    features = [np.nan_to_num(f) for f in features]

    #
    # random projection through a gaussian
    random_gaussians = [rand_gen.normal(size=(f.shape[1], k)) for f in features]

    rand_proj_features = [s / f.shape[1] * np.dot(f, N) for f, N in zip(features, random_gaussians)]

    nl_rand_proj_features = [non_linearity(f) for f in rand_proj_features]

    #
    # apply non-linearity
    if return_matrix:
        return np.concatenate(nl_rand_proj_features, axis=1)

    else:
        return [np.concatenate((f, np.ones((f.shape[0], 1))), axis=1) for f in nl_rand_proj_features]


def rdc_cca(indexes):
    i, j, rdc_features = indexes
    cca = CCA(n_components=1, max_iter=CCA_MAX_ITER)
    X_cca, Y_cca = cca.fit_transform(rdc_features[i], rdc_features[j])
    rdc = np.corrcoef(X_cca.T, Y_cca.T)[0, 1]
    # logger.info(i, j, rdc)
    return rdc


def rdc_test(local_data, meta_types, domains, k=None, s=1.0 / 6.0, non_linearity=np.sin, n_jobs=-1, rand_gen=None):
    n_features = local_data.shape[1]

    rdc_features = rdc_transformer(
        local_data, meta_types, domains, k=k, s=s, non_linearity=non_linearity, return_matrix=False, rand_gen=rand_gen
    )

    pairwise_comparisons = list(itertools.combinations(np.arange(n_features), 2))

    from joblib import Parallel, delayed

    rdc_vals = Parallel(n_jobs=n_jobs, max_nbytes=1024, backend="threading")(
        delayed(rdc_cca)((i, j, rdc_features)) for i, j in pairwise_comparisons
    )

    rdc_adjacency_matrix = np.zeros((n_features, n_features))

    for (i, j), rdc in zip(pairwise_comparisons, rdc_vals):
        rdc_adjacency_matrix[i, j] = rdc
        rdc_adjacency_matrix[j, i] = rdc

    #
    # setting diagonal to 1
    rdc_adjacency_matrix[np.diag_indices_from(rdc_adjacency_matrix)] = 1

    return rdc_adjacency_matrix


def getIndependentRDCGroups_py(
    local_data, threshold, meta_types, domains, k=None, s=1.0 / 6.0, non_linearity=np.sin, n_jobs=-2, rand_gen=None
):
    rdc_adjacency_matrix = rdc_test(
        local_data, meta_types, domains, k=k, s=s, non_linearity=non_linearity, n_jobs=n_jobs, rand_gen=rand_gen
    )

    # # print rdc_adjacency_matrix
    # print(f'rdc: {rdc_adjacency_matrix}')
    #
    # import pandas as pd
    # # 将local_data保存为csv
    # df = pd.DataFrame(local_data)
    # df.to_csv('/home/dafn/card/deepcard/imdb-benchmark/cup98_10/cup_98_10_test.csv')
    #
    # # 保存rdc_adjacency_matrix为csv
    # rdc_path = '/home/dafn/card/deepcard/imdb-benchmark/cup98_10/cup_98_10_test_rdc_adjacency_matrix.csv'
    # # 保存为CSV
    # rdc_df = pd.DataFrame(rdc_adjacency_matrix)
    # rdc_df.to_csv(rdc_path)


    #
    # Why is this necessary?
    #
    rdc_adjacency_matrix[np.isnan(rdc_adjacency_matrix)] = 0
    n_features = local_data.shape[1]

    ### for best threshold
    # import torch
    # from torch_geometric.data import Data, DataLoader
    # # import numpy as np
    # from Thresholding.gnn_regression import Config, MAMLModel, CorrelationDataset  # 根据实际路径导入
    #
    # def load_model(model_path, device):
    #     """加载训练好的模型并移动到指定设备"""
    #     checkpoint = torch.load(model_path, map_location=device)  # 使用指定设备加载
    #     config_dict = checkpoint['config']
    #     config = Config(**config_dict)
    #     model = MAMLModel(config, input_dim=11).to(device)  # 创建模型并移动到设备
    #     model.load_state_dict(checkpoint['state_dict'])
    #     model.eval()
    #     return model, config
    #
    # def numpy_to_data(adj_matrix: np.ndarray, threshold=0.5, config: Config = None, augment: bool = False,
    #                   device=None):
    #     adj_matrix = torch.as_tensor(adj_matrix, dtype=torch.float)
    #     # 处理负权重：转换为非负权重用于NetworkX的距离计算，保留原始权重用于GNN
    #     edge_weights_networkx = adj_matrix.abs()  # 非负权重（距离）
    #     edge_mask = edge_weights_networkx > 0  # 有效边为权重>0的边（非负）
    #
    #     if not edge_mask.any():
    #         raise ValueError(
    #             "Adjacency matrix must contain at least one positive weight after converting to non-negative.")
    #
    #     rows, cols = torch.where(edge_mask)
    #     edge_index = torch.stack([rows, cols], dim=0)
    #     edge_attr = adj_matrix[rows, cols]  # 原始边权重（可能包含负数，用于GNN）
    #     edge_distance = edge_weights_networkx[rows, cols]  # 非负距离权重（用于NetworkX）
    #
    #     N = adj_matrix.size(0)  # 获取节点数N
    #     # 将节点数作为特征(N,1)
    #     num_of_nodes = torch.arange(N).view(-1, 1).float()
    #
    #     # 排除对角线（自环）
    #     node_edges = adj_matrix.clone()
    #     node_edges.fill_diagonal_(0)
    #     all_edges = node_edges[~torch.eye(N, dtype=torch.bool)]  # 所有非对角线边
    #
    #     # ------------------- 全局权重统计（广播到节点） -------------------
    #     global_max = all_edges.max().view(1, 1).expand(N, 1)
    #     global_mean = all_edges.mean().view(1, 1).expand(N, 1)
    #     global_median = all_edges.median().view(1, 1).expand(N, 1)
    #     global_var = all_edges.var().view(1, 1).expand(N, 1)
    #     global_std = all_edges.std().view(1, 1).expand(N, 1)
    #
    #     # 随机采样N个权重
    #     random_selected_weights = all_edges[torch.randperm(all_edges.numel())[:N]].unsqueeze(1)
    #
    #     # ------------------- 节点级权重分布特征 -------------------
    #     node_mean = node_edges.mean(dim=1, keepdim=True)
    #     node_median = node_edges.median(dim=1, keepdim=True).values
    #     node_std = node_edges.std(dim=1, keepdim=True)
    #     node_var = node_edges.var(dim=1, keepdim=True)
    #     high_weight_threshold = 0.6  # 可调整
    #     node_high_ratio = (node_edges > high_weight_threshold).float().mean(dim=1, keepdim=True)
    #     low_weight_threshold = 0.4
    #     node_low_ratio = (node_edges < low_weight_threshold).float().mean(dim=1, keepdim=True)
    #
    #     # ------------------- 权重模式特征 -------------------
    #     node_deviation = (node_mean - global_mean) / (global_std + 1e-6)  # Z-score偏差
    #     node_25q = torch.quantile(node_edges, 0.25, dim=1, keepdim=True)
    #     node_75q = torch.quantile(node_edges, 0.75, dim=1, keepdim=True)
    #     node_iqr = node_75q - node_25q  # 四分位距
    #
    #     # =================== 步骤2：阈值处理邻接矩阵（新增关键逻辑） ================
    #     # 将低于0.5的边权重置零（形成稀疏图结构）
    #     thresholded_adj = adj_matrix.masked_fill(
    #         adj_matrix < config.feature_threshold,
    #         0.0
    #     )
    #     thresholded_adj.fill_diagonal_(0)  # 再次确保排除自环
    #
    #     # ------------------- 步骤3：转换为networkx图（用于结构特征计算） -------------------
    #     # 将阈值处理后的邻接矩阵转换为networkx无向图（假设是无向图，有向图需调整）
    #     import networkx as nx
    #
    #     G = nx.Graph()
    #     G.add_nodes_from(range(N))  # 添加节点
    #     # 添加边（仅保留权重>0的边）
    #     for i in range(N):
    #         for j in range(i + 1, N):  # 无向图避免重复边
    #             weight = thresholded_adj[i, j].item()
    #             # weight = adj_matrix[i, j].item()
    #             if weight > 0:
    #                 G.add_edge(i, j, weight=weight)
    #
    #     # ------------------- 步骤4：计算图结构特征（新增8维） -------------------
    #     # 1. 连通分量数（全局特征，广播到所有节点）
    #     connected_components = list(nx.connected_components(G))
    #     # 为每个连通分量分配唯一ID（从0开始）
    #     cc_mapping = {}
    #     for cc_id, cc in enumerate(connected_components):
    #         for node in cc:
    #             cc_mapping[node] = cc_id
    #
    #     # 为每个节点创建所属连通分量的特征
    #     cc_feature = torch.zeros(N, 1, dtype=torch.float)
    #     for node, cc_id in cc_mapping.items():
    #         cc_feature[node] = cc_id
    #
    #     # 可选：对连通分量ID进行归一化，使模型更好学习
    #     # if len(connected_components) > 1:
    #     #     cc_feature = cc_feature / (len(connected_components) - 1)  # 归一化到[0,1]区间
    #
    #     # 2. 节点度数（阈值处理后的度数，即与其他节点有效连接数）
    #     degrees = torch.tensor([d for _, d in G.degree()], dtype=torch.float).view(N, 1)  # [N,1]
    #
    #     # 3. 节点介数中心性（衡量节点作为桥梁的重要性）
    #     betweenness = nx.betweenness_centrality(G)  # 字典：{节点: 中心性值}
    #     betweenness_feature = torch.tensor([betweenness[i] for i in range(N)]).view(N, 1)  # [N,1]
    #
    #     # 4. 节点紧密中心性（衡量节点到其他节点的最短路径长度）
    #     closeness = nx.closeness_centrality(G)  # 字典：{节点: 中心性值}
    #     closeness_feature = torch.tensor([closeness[i] for i in range(N)]).view(N, 1)  # [N,1]
    #
    #     # 5. 节点特征向量中心性（衡量节点与高中心性节点的连接强度）
    #     try:
    #         eigenvector = nx.eigenvector_centrality(G, max_iter=1000)  # 可能需要增加迭代次数
    #     except nx.PowerIterationFailedConvergence:
    #         eigenvector = {i: 0.0 for i in range(N)}  # 收敛失败时置零
    #     eigenvector_feature = torch.tensor([eigenvector[i] for i in range(N)]).view(N, 1)  # [N,1]
    #
    #     # 6. 节点聚类系数（阈值处理后，邻居间的连接紧密程度）
    #     clustering = nx.clustering(G)  # 字典：{节点: 聚类系数}
    #     clustering_feature = torch.tensor([clustering[i] for i in range(N)]).view(N, 1)  # [N,1]
    #
    #     # 7. 平均路径长度（全局特征，仅在单连通分量时有效）
    #     if connected_components == 1:
    #         avg_path_length = nx.average_shortest_path_length(G)
    #     else:
    #         avg_path_length = -1.0  # 多连通分量时标记为-1
    #     avg_path_feature = torch.full((N, 1), avg_path_length, dtype=torch.float)  # [N,1]
    #
    #     # 8. 图密度（阈值处理后的边密度，0~1之间
    #     density = nx.density(G)
    #     density_feature = torch.full((N, 1), density, dtype=torch.float)  # [N,1]
    #
    #     # ------------------- 特征拼接与归一化 -------------------
    #     x = torch.cat(
    #         [density_feature, node_high_ratio, num_of_nodes, global_var, betweenness_feature, random_selected_weights, node_median, degrees, global_median, cc_feature, node_var], dim=1)
    #
    #     # ------------------- 特征标准化（跨图统一尺度） -------------------
    #     x = torch.nn.functional.normalize(x, p=2, dim=1)  # L2归一化
    #
    #     data = Data(
    #         x=x,
    #         edge_index=edge_index,
    #         edge_attr=edge_attr,  # 保留原始边权重（支持负数，用于GNN注意力）
    #         y=torch.tensor([threshold], dtype=torch.float)
    #     )
    #
    #     if device is not None:
    #         data = data.to(device)
    #
    #     return data
    #
    # def predict_threshold(model, config, adj_matrix):
    #     """预测输入邻接矩阵的阈值"""
    #     device = config.device
    #     data = numpy_to_data(adj_matrix, config=config, device=device)  # 传递设备参数
    #
    #     # print(f"data.x 设备: {data.x.device}")
    #     # print(f"data.edge_index 设备: {data.edge_index.device}")
    #     # print(f"data.edge_attr 设备: {data.edge_attr.device}")
    #
    #     with torch.no_grad():
    #         # 确保batch信息正确传递
    #         batch = torch.zeros(data.num_nodes, dtype=torch.int64, device=device)
    #         embedding = model.encoder(data.x, data.edge_index, data.edge_attr, batch)
    #         pred = model.predictor(embedding)
    #     return pred.item()
    #
    # # 加载模型部分
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_path = "./Thresholding/maml_threshold_model.pth"
    #
    # # 修正此处：确保 load_model 返回 (model, config)
    # model, config = load_model(model_path, device)
    # model.to(device)
    #
    # # 假设 rdc_adjacency_matrix 已定义
    # # 例如: rdc_adjacency_matrix = np.random.rand(10, 10)  # 示例随机矩阵
    #
    # predicted_threshold = predict_threshold(model, config, rdc_adjacency_matrix)
    # threshold = round(predicted_threshold, 4)
    # print(f"预测阈值为: {predicted_threshold:.4f}")
    # # 打印rdc_adjacency_matrix的形状
    # print(f"rdc_adjacency_matrix的形状为: {rdc_adjacency_matrix.shape}")


    ######    threshold predictor without meta learning and gat, purely mlp
    # import numpy as np
    import torch
    import networkx as nx
    import matplotlib.pyplot as plt

    # ----------------------------
    # 1) Config / Model definition
    # ----------------------------
    class Config:
        def __init__(
                self,
                lr=1e-4,
                batch_size=16,
                num_epochs=200,
                weight_decay=1e-5,
                device="cuda" if torch.cuda.is_available() else "cpu",
                hidden_dim=64,
                dropout_prob=0.2,
                feature_threshold=0.5,
                iter_seed=4303,
                loss_plot_path="mlp_loss_curve.png",
        ):
            self.lr = lr
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.weight_decay = weight_decay
            self.device = device
            self.hidden_dim = hidden_dim
            self.dropout_prob = dropout_prob
            self.feature_threshold = feature_threshold
            self.iter_seed = iter_seed
            self.loss_plot_path = loss_plot_path

    class MLPThresholdRegressor(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, dropout_prob: float):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout_prob),
                torch.nn.Linear(hidden_dim, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    # ----------------------------
    # 2) Feature extraction (same cost logic as MetaATPM baseline training)
    #    Note: output is graph-level feature vector [F]
    # ----------------------------
    def extract_graph_level_features(adj_matrix: np.ndarray, config: Config) -> torch.Tensor:
        adj_matrix = torch.as_tensor(adj_matrix, dtype=torch.float)
        N = adj_matrix.size(0)

        # If too small, return fixed-size vector (F=11 for the feature subset used below)
        if N <= 1:
            return torch.zeros(11, dtype=torch.float)

        # Remove diagonal for stats
        node_edges = adj_matrix.clone()
        node_edges.fill_diagonal_(0)
        all_edges = node_edges[~torch.eye(N, dtype=torch.bool)]

        if all_edges.numel() == 0:
            return torch.zeros(11, dtype=torch.float)

        # Global stats (broadcast)
        global_median = all_edges.median().view(1, 1).expand(N, 1)
        global_var = all_edges.var(unbiased=False).view(1, 1).expand(N, 1)

        # Random sample N edge weights (off-diagonal)
        perm = torch.randperm(all_edges.numel())
        sample_cnt = min(N, all_edges.numel())
        random_selected_weights = all_edges[perm[:sample_cnt]].view(-1, 1)
        if sample_cnt < N:
            pad = torch.zeros((N - sample_cnt, 1), dtype=torch.float)
            random_selected_weights = torch.cat([random_selected_weights, pad], dim=0)

        # Node-level stats
        node_var = node_edges.var(dim=1, keepdim=True, unbiased=False)
        node_median = node_edges.median(dim=1, keepdim=True).values
        high_weight_threshold = 0.6
        node_high_ratio = (node_edges > high_weight_threshold).float().mean(dim=1, keepdim=True)

        # Threshold for topology features
        thresholded_adj = adj_matrix.masked_fill(adj_matrix < config.feature_threshold, 0.0)
        thresholded_adj.fill_diagonal_(0)

        # Build undirected graph for topology features
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(i + 1, N):
                w = float(thresholded_adj[i, j].item())
                if w > 0:
                    G.add_edge(i, j, weight=w)

        # Connected components id per node
        connected_components = list(nx.connected_components(G))
        cc_mapping = {}
        for cc_id, cc in enumerate(connected_components):
            for node in cc:
                cc_mapping[node] = cc_id
        cc_feature = torch.zeros(N, 1, dtype=torch.float)
        for node, cc_id in cc_mapping.items():
            cc_feature[node] = float(cc_id)

        # Degree
        degrees = torch.tensor([d for _, d in G.degree()], dtype=torch.float).view(N, 1)

        # Betweenness
        betweenness = nx.betweenness_centrality(G)
        betweenness_feature = torch.tensor([betweenness[i] for i in range(N)], dtype=torch.float).view(N, 1)

        # Density (broadcast)
        density = nx.density(G)
        density_feature = torch.full((N, 1), float(density), dtype=torch.float)

        # Node count feature (arange)
        num_of_nodes = torch.arange(N, dtype=torch.float).view(-1, 1)

        # IMPORTANT: Keep the SAME feature list/order as your baseline training code
        x = torch.cat(
            [
                cc_feature,
                node_var,
                random_selected_weights,
                global_var,
                density_feature,
                global_median,
                betweenness_feature,
                node_median,
                node_high_ratio,
                degrees,
                num_of_nodes,
            ],
            dim=1,
        )

        # L2 normalize per node
        x = torch.nn.functional.normalize(x, p=2, dim=1)

        # Mean pool to graph-level feature
        graph_feat = x.mean(dim=0)  # [F]
        return graph_feat

    # ----------------------------
    # 3) Load trained baseline MLP model
    # ----------------------------
    def load_mlp_threshold_model(model_path: str, device: torch.device):
        """
        checkpoint format (from baseline training):
          {
            'state_dict': ...,
            'config': dict(...),
            'losses': [...],
            'input_dim': int,
            ...
          }
        """
        checkpoint = torch.load(model_path, map_location=device)

        config_dict = checkpoint.get("config", {})
        config = Config(**config_dict)
        # Force device to match caller
        config.device = str(device)

        input_dim = checkpoint.get("input_dim", 11)
        model = MLPThresholdRegressor(
            input_dim=input_dim,
            hidden_dim=getattr(config, "hidden_dim", 64),
            dropout_prob=getattr(config, "dropout_prob", 0.2),
        ).to(device)

        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        losses = checkpoint.get("losses", None)
        return model, config, losses

    # ----------------------------
    # 4) Optional: plot loss curve if stored in checkpoint
    # ----------------------------
    def plot_loss_curve(losses, save_path="mlp_loss_curve_loaded.png"):
        if losses is None or len(losses) == 0:
            print("[INFO] No stored losses in checkpoint; skip plotting.")
            return
        plt.figure()
        plt.plot(range(1, len(losses) + 1), losses)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training Loss Curve (Loaded)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[INFO] Loss curve saved to: {save_path}")

    # ----------------------------
    # 5) Predict threshold
    # ----------------------------
    @torch.no_grad()
    def predict_threshold(model, config: Config, adj_matrix: np.ndarray) -> float:
        device = torch.device(config.device) if isinstance(config.device, str) else config.device
        feat = extract_graph_level_features(adj_matrix, config).to(device)  # [F]
        pred = model(feat.unsqueeze(0))  # [1,1]
        return float(pred.item())

    # ----------------------------
    # 6) Example usage (same style as your original snippet)
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./Thresholding/baseline_mlp_threshold_model.pth"  # <-- update if needed

    model, config, losses = load_mlp_threshold_model(model_path, device)
    # optional: plot stored loss curve
    plot_loss_curve(losses, save_path="mlp_loss_curve_loaded.png")

    # rdc_adjacency_matrix must be a numpy.ndarray (NxN)
    # rdc_adjacency_matrix = np.random.rand(10, 10).astype(np.float32)  # example
    # rdc_adjacency_matrix = (rdc_adjacency_matrix + rdc_adjacency_matrix.T) / 2
    # np.fill_diagonal(rdc_adjacency_matrix, 1.0)

    predicted_threshold = predict_threshold(model, config, rdc_adjacency_matrix)
    threshold = round(predicted_threshold, 4)
    print(f"预测阈值为: {predicted_threshold:.4f}")
    print(f"rdc_adjacency_matrix的形状为: {rdc_adjacency_matrix.shape}")


    #
    # thresholding
    rdc_adjacency_matrix[rdc_adjacency_matrix < threshold] = 0
    # logger.info("thresholding %s", rdc_adjacency_matrix)

    #
    # getting connected components
    result = np.zeros(n_features)
    for i, c in enumerate(connected_components(from_numpy_matrix(rdc_adjacency_matrix))):
        result[list(c)] = i + 1

    return result


def getIndependentRDCGroups_py_test(
    local_data, rdc_threshold_low, meta_types, domains, k=None, s=1.0 / 6.0, non_linearity=np.sin, n_jobs=-2, rand_gen=None,
        rdc_threshold_high=0.5
):
    rdc_adjacency_matrix = rdc_test(
        local_data, meta_types, domains, k=k, s=s, non_linearity=non_linearity, n_jobs=n_jobs, rand_gen=rand_gen
    )

    #
    # Why is this necessary?
    #
    rdc_adjacency_matrix[np.isnan(rdc_adjacency_matrix)] = 0
    n_features = local_data.shape[1]

    result = np.zeros(n_features)
    i = 1
    # group_dicy: mark different groups, eg:highly correlated
    group_dict = {}
    # dependence_dict: mark condition for medium correlated groups
    dependence_dict = {}

    ### for best threshold
    import torch
    from torch_geometric.data import Data, DataLoader
    # import numpy as np
    # from Thresholding.gnn_regression import Config, MAMLModel, CorrelationDataset  # 根据实际路径导入
    from Thresholding.gnn_regression_double_threshold import Config, MAMLModel, CorrelationDataset

    def load_model(model_path, device):
        """加载训练好的模型并移动到指定设备"""
        checkpoint = torch.load(model_path, map_location=device)  # 使用指定设备加载
        config_dict = checkpoint['config']
        # 过滤掉Config类中不存在的参数
        config_fields = [f.name for f in Config.__dataclass_fields__.values()]
        filtered_config_dict = {k: v for k, v in config_dict.items() if k in config_fields}
        config = Config(**filtered_config_dict)
        model = MAMLModel(config, input_dim=11).to(device)  # 创建模型并移动到设备
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model, config

    def numpy_to_data(adj_matrix: np.ndarray, threshold=0.5, config: Config = None, augment: bool = False,
                      device=None):
        adj_matrix = torch.as_tensor(adj_matrix, dtype=torch.float)
        # 处理负权重：转换为非负权重用于NetworkX的距离计算，保留原始权重用于GNN
        edge_weights_networkx = adj_matrix.abs()  # 非负权重（距离）
        edge_mask = edge_weights_networkx > 0  # 有效边为权重>0的边（非负）

        if not edge_mask.any():
            raise ValueError(
                "Adjacency matrix must contain at least one positive weight after converting to non-negative.")

        rows, cols = torch.where(edge_mask)
        edge_index = torch.stack([rows, cols], dim=0)
        edge_attr = adj_matrix[rows, cols]  # 原始边权重（可能包含负数，用于GNN）
        edge_distance = edge_weights_networkx[rows, cols]  # 非负距离权重（用于NetworkX）

        N = adj_matrix.size(0)  # 获取节点数N
        # 将节点数作为特征(N,1)
        num_of_nodes = torch.arange(N).view(-1, 1).float()

        # 排除对角线（自环）
        node_edges = adj_matrix.clone()
        node_edges.fill_diagonal_(0)
        all_edges = node_edges[~torch.eye(N, dtype=torch.bool)]  # 所有非对角线边

        # ------------------- 全局权重统计（广播到节点） -------------------
        global_max = all_edges.max().view(1, 1).expand(N, 1)
        global_mean = all_edges.mean().view(1, 1).expand(N, 1)
        global_median = all_edges.median().view(1, 1).expand(N, 1)
        global_var = all_edges.var().view(1, 1).expand(N, 1)
        global_std = all_edges.std().view(1, 1).expand(N, 1)

        # 随机采样N个权重
        random_selected_weights = all_edges[torch.randperm(all_edges.numel())[:N]].unsqueeze(1)

        # ------------------- 节点级权重分布特征 -------------------
        node_mean = node_edges.mean(dim=1, keepdim=True)
        node_median = node_edges.median(dim=1, keepdim=True).values
        node_std = node_edges.std(dim=1, keepdim=True)
        node_var = node_edges.var(dim=1, keepdim=True)
        high_weight_threshold = 0.6  # 可调整
        node_high_ratio = (node_edges > high_weight_threshold).float().mean(dim=1, keepdim=True)
        low_weight_threshold = 0.4
        node_low_ratio = (node_edges < low_weight_threshold).float().mean(dim=1, keepdim=True)

        # ------------------- 权重模式特征 -------------------
        node_deviation = (node_mean - global_mean) / (global_std + 1e-6)  # Z-score偏差
        node_25q = torch.quantile(node_edges, 0.25, dim=1, keepdim=True)
        node_75q = torch.quantile(node_edges, 0.75, dim=1, keepdim=True)
        node_iqr = node_75q - node_25q  # 四分位距

        # =================== 步骤2：阈值处理邻接矩阵（新增关键逻辑） ================
        # 将低于0.5的边权重置零（形成稀疏图结构）
        thresholded_adj = adj_matrix.masked_fill(
            adj_matrix < config.feature_threshold,
            0.0
        )
        thresholded_adj.fill_diagonal_(0)  # 再次确保排除自环

        # ------------------- 步骤3：转换为networkx图（用于结构特征计算） -------------------
        # 将阈值处理后的邻接矩阵转换为networkx无向图（假设是无向图，有向图需调整）
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(N))  # 添加节点
        # 添加边（仅保留权重>0的边）
        for i in range(N):
            for j in range(i + 1, N):  # 无向图避免重复边
                weight = thresholded_adj[i, j].item()
                # weight = adj_matrix[i, j].item()
                if weight > 0:
                    G.add_edge(i, j, weight=weight)

        # ------------------- 步骤4：计算图结构特征（新增8维） -------------------
        # 1. 连通分量数（全局特征，广播到所有节点）
        connected_components = list(nx.connected_components(G))
        # 为每个连通分量分配唯一ID（从0开始）
        cc_mapping = {}
        for cc_id, cc in enumerate(connected_components):
            for node in cc:
                cc_mapping[node] = cc_id

        # 为每个节点创建所属连通分量的特征
        cc_feature = torch.zeros(N, 1, dtype=torch.float)
        for node, cc_id in cc_mapping.items():
            cc_feature[node] = cc_id

        # 可选：对连通分量ID进行归一化，使模型更好学习
        # if len(connected_components) > 1:
        #     cc_feature = cc_feature / (len(connected_components) - 1)  # 归一化到[0,1]区间

        # 2. 节点度数（阈值处理后的度数，即与其他节点有效连接数）
        degrees = torch.tensor([d for _, d in G.degree()], dtype=torch.float).view(N, 1)  # [N,1]

        # 3. 节点介数中心性（衡量节点作为桥梁的重要性）
        betweenness = nx.betweenness_centrality(G)  # 字典：{节点: 中心性值}
        betweenness_feature = torch.tensor([betweenness[i] for i in range(N)]).view(N, 1)  # [N,1]

        # 4. 节点紧密中心性（衡量节点到其他节点的最短路径长度）
        closeness = nx.closeness_centrality(G)  # 字典：{节点: 中心性值}
        closeness_feature = torch.tensor([closeness[i] for i in range(N)]).view(N, 1)  # [N,1]

        # 5. 节点特征向量中心性（衡量节点与高中心性节点的连接强度）
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000)  # 可能需要增加迭代次数
        except nx.PowerIterationFailedConvergence:
            eigenvector = {i: 0.0 for i in range(N)}  # 收敛失败时置零
        eigenvector_feature = torch.tensor([eigenvector[i] for i in range(N)]).view(N, 1)  # [N,1]

        # 6. 节点聚类系数（阈值处理后，邻居间的连接紧密程度）
        clustering = nx.clustering(G)  # 字典：{节点: 聚类系数}
        clustering_feature = torch.tensor([clustering[i] for i in range(N)]).view(N, 1)  # [N,1]

        # 7. 平均路径长度（全局特征，仅在单连通分量时有效）
        if connected_components == 1:
            avg_path_length = nx.average_shortest_path_length(G)
        else:
            avg_path_length = -1.0  # 多连通分量时标记为-1
        avg_path_feature = torch.full((N, 1), avg_path_length, dtype=torch.float)  # [N,1]

        # 8. 图密度（阈值处理后的边密度，0~1之间
        density = nx.density(G)
        density_feature = torch.full((N, 1), density, dtype=torch.float)  # [N,1]

        # ------------------- 特征拼接与归一化 -------------------
        x = torch.cat(
            [global_median, betweenness_feature, global_var, node_var, density_feature, num_of_nodes, cc_feature, node_high_ratio, random_selected_weights, node_median, degrees], dim=1)

        # ------------------- 特征标准化（跨图统一尺度） -------------------
        x = torch.nn.functional.normalize(x, p=2, dim=1)  # L2归一化

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,  # 保留原始边权重（支持负数，用于GNN注意力）
            y=torch.tensor([threshold], dtype=torch.float)
        )

        if device is not None:
            data = data.to(device)

        return data

    def predict_threshold(model, config, adj_matrix):
        """预测输入邻接矩阵的阈值"""
        device = config.device
        data = numpy_to_data(adj_matrix, config=config, device=device)  # 传递设备参数

        # print(f"data.x 设备: {data.x.device}")
        # print(f"data.edge_index 设备: {data.edge_index.device}")
        # print(f"data.edge_attr 设备: {data.edge_attr.device}")

        with torch.no_grad():
            # 确保batch信息正确传递
            batch = torch.zeros(data.num_nodes, dtype=torch.int64, device=device)
            embedding = model.encoder(data.x, data.edge_index, data.edge_attr, batch)
            pred = model.predictor(embedding)
        return pred.cpu().numpy()

    # 加载模型部分
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./Thresholding/maml_threshold_model.pth"

    # 修正此处：确保 load_model 返回 (model, config)
    model, config = load_model(model_path, device)
    model.to(device)

    # 假设 rdc_adjacency_matrix 已定义
    # 例如: rdc_adjacency_matrix = np.random.rand(10, 10)  # 示例随机矩阵

    predicted_threshold = predict_threshold(model, config, rdc_adjacency_matrix)
    rdc_threshold_low = round(predicted_threshold[0][0], 4)
    rdc_threshold_high = round(predicted_threshold[0][1], 4)
    print(f"预测阈值为: {rdc_threshold_low:.4f}, {rdc_threshold_high:.4f}")
    # 打印rdc_adjacency_matrix的形状
    print(f"rdc_adjacency_matrix的形状为: {rdc_adjacency_matrix.shape}")


    ####
    # thresholding for high correlated groups
    rdc_adjacency_matrix_high = rdc_adjacency_matrix.copy()
    rdc_adjacency_matrix_high[rdc_adjacency_matrix_high < rdc_threshold_high] = 0
    # logger.info("thresholding %s", rdc_adjacency_matrix)
    #
    # getting connected components

    for _, c in enumerate(connected_components(from_numpy_matrix(rdc_adjacency_matrix_high))):
        if len(list(c)) > 1:
            result[list(c)] = i
            # if group_dict.get("highly correlated") is not None:
            #     group_dict["highly correlated"].append(i)
            # else:
            #     group_dict["highly correlated"] = [i]
            i += 1

    ###
    # thresholding for medium correlated groups
    rdc_adjacency_matrix_medium = rdc_adjacency_matrix.copy()
    rdc_adjacency_matrix_medium[rdc_adjacency_matrix_medium < rdc_threshold_low] = 0
    rdc_adjacency_matrix_medium[rdc_adjacency_matrix_medium > rdc_threshold_high] = 0

    # getting connected components
    for _, c in enumerate(connected_components(from_numpy_matrix(rdc_adjacency_matrix_medium))):
        temp = [j for j in list(c) if result[j] == 0]
        if len(temp) > 0:
            result[temp] = i
            i += 1
        # elif len(temp) == 1:
        # conditional split. if change here, the whole process logic should be revised because temp here only contains
        # one col and will be treated as a histgram.


    ###y correlate
    # thresholding for weakld groups
    rdc_adjacency_matrix_weak = rdc_adjacency_matrix.copy()
    rdc_adjacency_matrix_weak[rdc_threshold_low < rdc_adjacency_matrix_weak] = 0

    # getting connected components
    for _, c in enumerate(connected_components(from_numpy_matrix(rdc_adjacency_matrix_weak))):
        # if len(c) == 1:
        #     c = list(c)
        #     if result[c[0]] == 0:
        #         result[c[0]] = i
        #         i += 1
        temp = [j for j in list(c) if result[j] == 0]
        if len(temp) > 0:
            result[temp] = i
            i += 1

    for j in range(n_features):
        if result[j] == 0:
            result[j] = i
            i += 1

    return result, group_dict, dependence_dict



def get_split_cols_RDC_py(threshold=0.3, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None):
    def split_cols_RDC_py(local_data, ds_context, scope):
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        # clusters = getIndependentRDCGroups_py(
        clusters = getIndependentRDCGroups_py(
            local_data,
            threshold,
            meta_types,
            domains,
            k=k,
            s=s,
            # ohe=True,
            non_linearity=non_linearity,
            n_jobs=n_jobs,
            rand_gen=rand_gen,
        )

        ### mark
        # change this split func, add conditions
        return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_RDC_py


def get_split_rows_RDC_py(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None):
    def split_rows_RDC_py(local_data, ds_context, scope):
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        rdc_data = rdc_transformer(
            local_data,
            meta_types,
            domains,
            k=k,
            s=s,
            non_linearity=non_linearity,
            return_matrix=True,
            rand_gen=rand_gen,
        )

        clusters = KMeans(n_clusters=n_clusters, random_state=rand_gen, n_jobs=n_jobs).fit_predict(rdc_data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_RDC_py
