import os
import numpy as np
import pandas as pd

def load_matrices_and_generate_labels(main_folder):
    data = []
    labels = []

    # 正确读取CSV（假设第一行为列名）
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'best_thresholds_single_threshold.csv'), header=0)
    df.set_index('Table Name', inplace=True)  # 以Table Name为索引

    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith('matrix.csv'):
                matrix_file = os.path.join(root, file)
                try:
                    matrix = pd.read_csv(matrix_file, header=None).values
                    matrix = np.delete(matrix, 0, axis=0)
                    matrix = np.delete(matrix, 0, axis=1)
                    data.append(matrix)

                    # 获取文件名并匹配标签
                    table_name = os.path.splitext(file)[0].split('_rdc_adjacency_matrix')[0]
                    if table_name in df.index:
                        label = df.loc[table_name, 'Best RDC Threshold']
                        labels.append(label)
                    else:
                        raise KeyError(f"未找到表格名 '{table_name}' 对应的阈值")
                except Exception as e:
                    print(f"处理文件 {matrix_file} 时出错: {str(e)}")

    print(f"加载的矩阵数量: {len(data)}, 标签数量: {len(labels)}")
    return data, labels

if __name__ == "__main__":
    main_folder = "/home/dafn/card/deepcard/collected_datasets"
    data, labels = load_matrices_and_generate_labels(main_folder)
