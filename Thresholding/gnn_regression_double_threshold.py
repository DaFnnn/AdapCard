import torch
import torch_geometric.nn as geom_nn
from dataclasses import dataclass
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import higher
from typing import List, Optional, Union 

@dataclass
class Config:
    gnn_hidden_dim: int = 16
    meta_lr: float = 1e-5
    adaptation_steps: int = 1
    adaptation_lr: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 220
    weight_decay: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    dropout_prob: float = 0.5
    predictor_hidden_dim: int = 32
    num_of_heads: int = 2

    feature_threshold: float = 0.5

    iter_seed: int = 974


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
            torch.nn.Linear(hidden_dim, 2)   # 输出两个阈值
        )

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        return self.fc(graph_embedding)  # shape [batch_size, 2]


class MAMLModel(torch.nn.Module):
    def __init__(self, config: Config, input_dim: int):
        super().__init__()
        self.config = config
        self.encoder = GNNEncoder(input_dim, config.gnn_hidden_dim, config.dropout_prob, config.num_of_heads)
        self.predictor = ThresholdPredictor(
            input_dim=config.gnn_hidden_dim * config.num_of_heads,
            hidden_dim=config.predictor_hidden_dim
        )

    def forward(self, data: Data, params: Optional[dict] = None) -> torch.Tensor:
        if params is None:
            embedding = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
            return self.predictor(embedding)

        # 1. 按参数名前缀分割 encoder/predictor 的参数（避免列表切片）
        encoder_params = {k: v for k, v in params.items() if k.startswith('encoder.')}
        predictor_params = {k: v for k, v in params.items() if k.startswith('predictor.')}

        # 2. 移除参数名前缀（匹配子模块的原始参数名，如 "encoder.conv1.weight" → "conv1.weight"）
        encoder_params = {k.replace('encoder.', '', 1): v for k, v in encoder_params.items()}
        predictor_params = {k.replace('predictor.', '', 1): v for k, v in predictor_params.items()}

        # 3. 用字典格式传递参数给 patch_module
        with higher.patch_module(self.encoder, params=encoder_params) as patched_encoder:
            with higher.patch_module(self.predictor, params=predictor_params) as patched_predictor:
                embedding = patched_encoder(data.x, data.edge_index, data.edge_attr, data.batch)
                return patched_predictor(embedding)

    def _compute_loss(self, data: Data, params: Optional[dict] = None) -> torch.Tensor:
        pred = self.forward(data, params)  # shape [batch_size, 2]
        return torch.nn.functional.mse_loss(pred, data.y.float())


class CorrelationDataset(Dataset):
    def __init__(self, matrices: List[torch.Tensor], labels: List[Union[float, List[float], tuple, torch.Tensor]],
                 input_dim: int = None, augment: bool = False,
                 config: Config = None):
        super().__init__()
        self.data_list = []
        self.augment = augment
        self.input_dim = input_dim
        self.config = config if config else Config()

        # 简单检查：labels 长度应与 matrices 一致，若 labels 为单个值则广播
        if labels is None:
            # 如果没有标签，使用默认(0.5,0.5)
            labels = [(0.5, 0.5)] * len(matrices)
        elif len(labels) == 1 and len(matrices) > 1:
            # 单标签 -> 广播
            labels = [labels[0]] * len(matrices)
        elif len(labels) != len(matrices):
            # 不匹配则报错（避免静默截断）
            raise ValueError(f"Number of labels ({len(labels)}) does not match number of matrices ({len(matrices)}).")

        # 内部函数：把各种输入格式标准化为 (t1, t2) Python tuple
        def _normalize_label(label) -> tuple:
            # 支持 float/int, list/tuple, torch.Tensor
            if isinstance(label, (float, int)):
                return (float(label), float(label))
            if isinstance(label, torch.Tensor):
                cnt = label.numel()
                if cnt == 1:
                    return (float(label.item()), float(label.item()))
                elif cnt == 2:
                    a = float(label.view(-1)[0].item()); b = float(label.view(-1)[1].item())
                    return (a, b)
                else:
                    raise ValueError("Tensor label must have 1 or 2 elements.")
            if isinstance(label, (list, tuple)):
                if len(label) == 1:
                    return (float(label[0]), float(label[0]))
                if len(label) == 2:
                    return (float(label[0]), float(label[1]))
                raise ValueError("List/Tuple label must have length 1 or 2.")
            raise TypeError("Unsupported label type. Expect float, list/tuple, or torch.Tensor")

        for mat, label in zip(matrices, labels):
            label_pair = _normalize_label(label)
            graph = self.matrix_to_graph(mat, label_pair)
            self.data_list.append(graph)
            if self.input_dim is None:
                self.input_dim = graph.x.size(1)

    def matrix_to_graph(self, adj_matrix: torch.Tensor, thresholds: Union[float, List[float], tuple, torch.Tensor]) -> Data:
        # thresholds 在 __init__ 已被标准化成 (t1, t2) tuple，但这里仍做保险处理
        if isinstance(thresholds, (list, tuple, torch.Tensor)):
            t = torch.tensor(thresholds, dtype=torch.float)
            if t.numel() == 1:
                t = t.repeat(2)
            elif t.numel() != 2:
                raise ValueError("Threshold must be scalar or pair of scalars.")
        else:
            # scalar
            t = torch.tensor([float(thresholds), float(thresholds)], dtype=torch.float)

        # 将 t 变为 2D：shape (1, 2) —— 关键：保证每个样本的 y 是 (1,2)，这样 Batch 后得到 (batch_size, 2)
        y_tensor = t.view(1, -1)

        adj_matrix = torch.as_tensor(adj_matrix, dtype=torch.float)
        # 处理负权重：转换为非负权重用于NetworkX的距离计算，保留原始权重用于GNN
        edge_weights_networkx = adj_matrix.abs()  # 非负权重（距离）
        edge_mask = edge_weights_networkx > 0  # 有效边为权重>0的边（非负）

        if not edge_mask.any():
            raise ValueError(
                "Adjacency matrix must contain at least one positive weight after converting to non-negative.")

        rows, cols = torch.where(edge_mask)
        edge_index = torch.stack([rows, cols], dim=0)
        edge_attr = adj_matrix[rows, cols].unsqueeze(1).float()  # 原始边权重（可能包含负数，用于GNN）
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
        thresholded_adj = adj_matrix.masked_fill(
            adj_matrix < self.config.feature_threshold,
            0.0
        )
        thresholded_adj.fill_diagonal_(0)  # 再次确保排除自环

        # ------------------- 步骤3：转换为networkx图（用于结构特征计算） -------------------
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(N))  # 添加节点
        # 添加边（仅保留权重>0的边）
        for i in range(N):
            for j in range(i + 1, N):  # 无向图避免重复边
                weight = thresholded_adj[i, j].item()
                if weight > 0:
                    G.add_edge(i, j, weight=weight)

        if G.number_of_edges() == 0:
            print(f"Warning: Thresholded graph is empty, using original adjacency matrix (no threshold)")
            # 用原始邻接矩阵（不阈值）重新添加边
            for i in range(N):
                for j in range(i + 1, N):
                    weight = adj_matrix[i, j].item()
                    if weight != 0:  # 非零权重即添加
                        G.add_edge(i, j, weight=weight)
            # 若仍空图，抛出明确错误
            if G.number_of_edges() == 0:
                raise ValueError(f"Adjacency matrix {adj_matrix} has no non-zero edges (cannot create graph)")

        # ------------------- 步骤4：计算图结构特征（新增8维） -------------------
        connected_components = list(nx.connected_components(G))
        cc_mapping = {}
        for cc_id, cc in enumerate(connected_components):
            for node in cc:
                cc_mapping[node] = cc_id

        cc_feature = torch.zeros(N, 1, dtype=torch.float)
        for node, cc_id in cc_mapping.items():
            cc_feature[node] = cc_id

        degrees = torch.tensor([d for _, d in G.degree()], dtype=torch.float).view(N, 1)  # [N,1]

        betweenness = nx.betweenness_centrality(G)  # 字典：{节点: 中心性值}
        betweenness_feature = torch.tensor([betweenness[i] for i in range(N)]).view(N, 1)  # [N,1]

        closeness = nx.closeness_centrality(G)  # 字典：{节点: 中心性值}
        closeness_feature = torch.tensor([closeness[i] for i in range(N)]).view(N, 1)  # [N,1]

        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000)  # 可能需要增加迭代次数
        except nx.PowerIterationFailedConvergence:
            eigenvector = {i: 0.0 for i in range(N)}  # 收敛失败时置零
        eigenvector_feature = torch.tensor([eigenvector[i] for i in range(N)]).view(N, 1)  # [N,1]

        clustering = nx.clustering(G)  # 字典：{节点: 聚类系数}
        clustering_feature = torch.tensor([clustering[i] for i in range(N)]).view(N, 1)  # [N,1]

        if len(connected_components) == 1:
            try:
                avg_path_length = nx.average_shortest_path_length(G)
            except:
                avg_path_length = -1.0
        else:
            avg_path_length = -1.0  # 多连通分量时标记为-1
        avg_path_feature = torch.full((N, 1), avg_path_length, dtype=torch.float)  # [N,1]

        density = nx.density(G)
        density_feature = torch.full((N, 1), density, dtype=torch.float)  # [N,1]

        # ------------------- 特征拼接与归一化 -------------------
        x = torch.cat([global_median, betweenness_feature, global_var, node_var, density_feature, num_of_nodes, cc_feature, node_high_ratio, random_selected_weights, node_median, degrees], dim=1)

        # ------------------- 特征标准化（跨图统一尺度） -------------------
        x = torch.nn.functional.normalize(x, p=2, dim=1)  # L2归一化

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_tensor  # shape (1,2) per graph -> Batch 后为 (batch_size, 2)
        )

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

    def get_input_dim(self) -> int:
        return self.input_dim


def meta_train(model: MAMLModel, train_dataset: CorrelationDataset, config: Config) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.meta_lr, weight_decay=config.weight_decay)
    epoch_losses = []

    # 提前创建DataLoader（避免重复初始化）
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        pin_memory=(config.device == "cuda"),
        shuffle=True,
        persistent_workers=False
    )

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(loader):
            with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False, device=config.device) as (fnet, diffopt):
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

        # 关键修改：除以 batch 数量（len(loader)），而非样本数
        avg_epoch_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{config.num_epochs}], Average Loss: {avg_epoch_loss:.4f}")
        epoch_losses.append(avg_epoch_loss)

    return epoch_losses

def evaluate(model: MAMLModel, test_dataset: CorrelationDataset, config: Config) -> float:
    model.eval()
    loader = DataLoader(test_dataset, batch_size=config.batch_size)
    total_mae = torch.tensor(0.0, device=config.device)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(config.device)
            embedding = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = model.predictor(embedding)  # [batch, 2]
            print(f"predicted thresholds: {pred.detach().cpu().numpy()}")
            # batch.y 的形状现在应为 [batch, 2]
            total_mae += torch.abs(pred - batch.y).sum()
    # 平均 MAE（按元素平均 —— 每个图有两个阈值）
    return (total_mae / (len(test_dataset) * 2)).item()



import matplotlib.pyplot as plt

def plot_losses(losses: List[float], save_path: str = "loss_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss', linewidth=2)
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
    # 固定随机种子
    import random
    config = Config()
    seed = config.iter_seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
    import numpy as np
    np.random.seed(seed)

    from Thresholding.load_data_and_labels_double_threshold import load_matrices_and_generate_labels_double_threshold


    print("load training data")
    train_matrices, train_labels = load_matrices_and_generate_labels_double_threshold('/home/dafn/card/deepcard/collected_datasets')
    train_matrices = [torch.tensor(mat) for mat in train_matrices]

    print("load test data")
    test_matrices, _ = load_matrices_and_generate_labels_double_threshold('/home/dafn/card/deepcard/imdb-benchmark/forest')
    test_labels = [(0.25, 0.4), (0.25, 0.3)]
    test_matrices = [torch.tensor(mat) for mat in test_matrices]


    train_dataset = CorrelationDataset(train_matrices, train_labels, config=config)
    test_dataset = CorrelationDataset(test_matrices, test_labels, config=config)

    # print("训练标签前5个：", train_labels[:5])

    input_dim = train_dataset.get_input_dim()
    print(f"Input dimension: {input_dim}")
    model = MAMLModel(config, input_dim).to(config.device)

    try:
        losses = meta_train(model, train_dataset, config)
        mae = evaluate(model, test_dataset, config)
        print(f"Test MAE: {mae:.4f}")

        plot_losses(losses, "training_loss_curve.png")

        torch.save({
            "state_dict": model.state_dict(),
            "config": vars(config),
            "losses": losses
        }, "maml_threshold_model.pth")
    except Exception as e:
        print(f"Training error: {str(e)}")

if __name__ == "__main__":
    main()
