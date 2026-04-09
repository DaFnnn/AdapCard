import torch
import torch_geometric.nn as geom_nn
from dataclasses import dataclass
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import higher
from typing import List, Optional
import networkx as nx
import sys

sys.path.append("..")

@dataclass
class Config:
    gnn_hidden_dim: int = 32
    meta_lr: float = 1e-5
    adaptation_steps: int = 1
    adaptation_lr: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 245
    weight_decay: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    dropout_prob: float = 0.5
    predictor_hidden_dim: int = 32
    num_of_heads: int = 1

    feature_threshold: float = 0.5

    iter_seed: int = 4303


class GNNEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout_prob: float, num_of_heads: int):
        super().__init__()
        self.conv1 = geom_nn.GATConv(input_dim, hidden_dim, edge_dim=1, heads=num_of_heads, dropout=dropout_prob)
        self.conv2 = geom_nn.GATConv(hidden_dim * num_of_heads, hidden_dim, edge_dim=1, heads=num_of_heads, dropout=dropout_prob)
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.pool = geom_nn.global_mean_pool

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.dropout(torch.relu(x))
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return self.pool(x, batch)


class ThresholdPredictor(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        return self.fc(graph_embedding).squeeze(1)


class MAMLModel(torch.nn.Module):
    def __init__(self, config: Config, input_dim: int):
        super().__init__()
        self.config = config
        self.encoder = GNNEncoder(input_dim, config.gnn_hidden_dim, config.dropout_prob, config.num_of_heads)
        self.predictor = ThresholdPredictor(
            input_dim=config.gnn_hidden_dim * config.num_of_heads,
            hidden_dim=config.predictor_hidden_dim
        )

    def forward(self, data: Data, params: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if params is None:
            # 直接使用模型的原始参数
            embedding = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
            return self.predictor(embedding)

        # 获取 encoder 和 predictor 的命名参数
        encoder_named_params = list(self.encoder.named_parameters())
        predictor_named_params = list(self.predictor.named_parameters())
        num_encoder_params = len(encoder_named_params)

        # 分割参数为 encoder 和 predictor 的命名字典
        encoder_params = {
            name: param for (name, _), param in zip(encoder_named_params, params[:num_encoder_params])
        }
        predictor_params = {
            name: param for (name, _), param in zip(predictor_named_params, params[num_encoder_params:])
        }

        # 用命名参数字典 patch 模型
        with higher.patch_module(self.encoder, params=encoder_params) as patched_encoder:
            with higher.patch_module(self.predictor, params=predictor_params) as patched_predictor:
                embedding = patched_encoder(data.x, data.edge_index, data.edge_attr, data.batch)
                return patched_predictor(embedding)

    def _compute_loss(self, data: Data, params: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        pred = self.forward(data, params)
        return torch.nn.functional.mse_loss(pred, data.y.float())

class CorrelationDataset(Dataset):
    def __init__(self, matrices: List[torch.Tensor], labels: List[float],
                 input_dim: int = None, augment: bool = False,
                 config: Config = None):
        super().__init__()
        self.data_list = []
        # self.feature_order = feature_order
        self.augment = augment
        self.input_dim = input_dim
        self.config = config if config else Config()

        for mat, label in zip(matrices, labels):
            graph = self.matrix_to_graph(mat, label)
            self.data_list.append(graph)
            if self.input_dim is None:
                self.input_dim = graph.x.size(1)

    def matrix_to_graph(self, adj_matrix: torch.Tensor, threshold: float) -> Data:


        adj_matrix = torch.as_tensor(adj_matrix, dtype=torch.float)
        # 处理负权重：转换为非负权重用于NetworkX的距离计算，保留原始权重用于GNN
        edge_weights_networkx = adj_matrix.abs()  # 非负权重（距离）
        edge_mask = edge_weights_networkx > 0  # 有效边为权重>0的边（非负）

        if not edge_mask.any():
            raise ValueError(
                "Adjacency matrix must contain at least one positive weight after converting to non-negative.")

        rows, cols = torch.where(edge_mask)
        edge_index = torch.stack([rows, cols], dim=0)
        edge_attr = adj_matrix[rows, cols].unsqueeze(1)  # 原始边权重（可能包含负数，用于GNN）
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
        low_weight_threshold = 0.3
        node_low_ratio = (node_edges < low_weight_threshold).float().mean(dim=1, keepdim=True)

        # ------------------- 权重模式特征 -------------------
        node_deviation = (node_mean - global_mean) / (global_std + 1e-6)  # Z-score偏差
        node_25q = torch.quantile(node_edges, 0.25, dim=1, keepdim=True)
        node_75q = torch.quantile(node_edges, 0.75, dim=1, keepdim=True)
        node_iqr = node_75q - node_25q  # 四分位距

        # =================== 步骤2：阈值处理邻接矩阵（新增关键逻辑） ================
        # 将低于0.5的边权重置零（形成稀疏图结构）
        thresholded_adj = adj_matrix.masked_fill(
            adj_matrix < self.config.feature_threshold,
            0.0
        )
        thresholded_adj.fill_diagonal_(0)  # 再次确保排除自环

        # ------------------- 步骤3：转换为networkx图（用于结构特征计算） -------------------
        # 将阈值处理后的邻接矩阵转换为networkx无向图（假设是无向图，有向图需调整）

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
        if len(connected_components) == 1:  # 修正为判断列表长度
            avg_path_length = nx.average_shortest_path_length(G)
        else:
            avg_path_length = -1.0  # 多连通分量时标记为-1
        avg_path_feature = torch.full((N, 1), avg_path_length, dtype=torch.float)  # [N,1]

        # 8. 图密度（阈值处理后的边密度，0~1之间
        density = nx.density(G)
        density_feature = torch.full((N, 1), density, dtype=torch.float)  # [N,1]

        # ------------------- 特征拼接与归一化 -------------------
        x = torch.cat([cc_feature, node_var, random_selected_weights, global_var, density_feature, global_median, betweenness_feature, node_median, node_high_ratio, degrees, num_of_nodes], dim=1)

        # ------------------- 特征标准化（跨图统一尺度） -------------------
        x = torch.nn.functional.normalize(x, p=2, dim=1)  # L2归一化

        # # 数据增强（在返回Data对象前添加）
        # if augment:
        #     # 边权重扰动：添加高斯噪声
        #     edge_attr = edge_attr + torch.randn_like(edge_attr) * 0.1
        #     # 节点特征扰动：添加随机噪声
        #     x = x + torch.randn_like(x) * 0.1
        #     # 边删除增强：以5%概率随机删除边
        #     edge_mask = torch.rand(len(edge_index[0])) > 0.05
        #     edge_index = edge_index[:, edge_mask]
        #     edge_attr = edge_attr[edge_mask]

        return Data(
            x=x,
            edge_index=edge_index,
            # edge_attr=edge_attr,  # 保留原始边权重（支持负数，用于GNN注意力）
            y=torch.tensor([threshold], dtype=torch.float)
        )

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

    def get_input_dim(self) -> int:
        return self.input_dim


def meta_train(model: MAMLModel, train_dataset: CorrelationDataset, config: Config) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.meta_lr, weight_decay=config.weight_decay)
    epoch_losses = []  # 新增：记录每个epoch的损失

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(
                DataLoader(train_dataset, batch_size=config.batch_size, pin_memory=config.device == "cuda", shuffle=True)):
            with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (fnet, diffopt):
                batch = batch.to(config.device)
                for _ in range(config.adaptation_steps):
                    loss = fnet._compute_loss(batch)
                    diffopt.step(loss)
                meta_loss = fnet._compute_loss(batch)
                meta_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += meta_loss.item()

        avg_epoch_loss = epoch_loss / len(train_dataset)
        print(f"Epoch [{epoch + 1}/{config.num_epochs}], Average Loss: {avg_epoch_loss:.4f}")
        epoch_losses.append(avg_epoch_loss)  # 保存当前epoch损失

    return epoch_losses  # 返回所有epoch的损失记录

def evaluate(model: MAMLModel, test_dataset: CorrelationDataset, config: Config) -> float:
    model.eval()
    loader = DataLoader(test_dataset, batch_size=config.batch_size)
    total_mae = torch.tensor(0.0, device=config.device)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(config.device)
            embedding = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = model.predictor(embedding)
            # print(f"predicted threshold: {pred:.2f}")
            # 安全写法
            print(f"predicted threshold: {pred.detach().cpu().numpy()}")  # 显示完整内容
            total_mae += torch.abs(pred - batch.y).sum()
    return (total_mae / torch.tensor(len(test_dataset), device=config.device)).item()


import matplotlib.pyplot as plt

def plot_losses(losses: List[float], save_path: str = "loss_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss', color='blue', linewidth=2)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)  # 保存图像
    plt.show()  # 显示图像
    plt.close()


def main() -> None:
    # 新增：固定随机种子
    import random

    config = Config()

    seed = config.iter_seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        import torch.backends.cudnn as cudnn

        # 设置cuDNN为确定性模式
        cudnn.deterministic = True
        cudnn.benchmark = False

    import numpy as np

    np.random.seed(seed)

    from Thresholding.load_data_and_labels import load_matrices_and_generate_labels

    train_matrices, train_labels = load_matrices_and_generate_labels('../collected_datasets')
    train_matrices = [torch.tensor(mat) for mat in train_matrices]

    test_matrices, _ = load_matrices_and_generate_labels('../imdb-benchmark/forest')
    # temp threshold
    # test_labels = [0.4, 0.35, 0.3]
    test_labels = [0.3, 0.40] # 先读的census,（power）然后是forest
    test_matrices = [torch.tensor(mat) for mat in test_matrices]

    train_dataset = CorrelationDataset(train_matrices, train_labels, config=config)
    test_dataset = CorrelationDataset(test_matrices, test_labels, config=config)

    # ### for testing power
    # test_matrices_power, _ = load_matrices_and_generate_labels('/home/dafn/card/deepcard/imdb-benchmark/power')
    # test_label_power = [0.3]
    # test_matrices_power = [torch.tensor(mat) for mat in test_matrices_power]
    # test_dataset_power = CorrelationDataset(test_matrices_power, test_label_power, config=config)

    # ###
    print("训练标签前5个：", train_labels[:5])

    input_dim = train_dataset.get_input_dim()
    print(f"Input dimension: {input_dim}")
    model = MAMLModel(config, input_dim).to(config.device)

    try:
        import time
        start_time = time.time()
        losses = meta_train(model, train_dataset, config)  # 获取损失记录
        print(f"Time taken for training: {time.time() - start_time:.2f} seconds")
        mae = evaluate(model, test_dataset, config)
        print(f"Test MAE: {mae:.4f}")

        # ### for testing power
        # mae_power = evaluate(model, test_dataset_power, config)

        # 新增：绘制并保存损失曲线
        plot_losses(losses, "training_loss_curve.png")

        torch.save({
            "state_dict": model.state_dict(),
            "config": vars(config),
            "losses": losses  # 保存损失历史
        }, "maml_threshold_model.pth")
    except Exception as e:
        print(f"Training error: {str(e)}")

if __name__ == "__main__":
    main()
