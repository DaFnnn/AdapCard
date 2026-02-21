import os
import numpy as np
import pandas as pd


def str_to_tuple(threshold_str):
    """将形如'(0.3000, 0.6500)'的字符串转换为元组(0.3, 0.65)"""
    # 去除括号并分割成两个字符串
    values = threshold_str.strip('()').split(',')
    # 转换为浮点数并返回元组
    return tuple(float(value.strip()) for value in values)


def load_matrices_and_generate_labels_double_threshold(main_folder):
    data = []
    labels = []

    # 使用正则表达式作为分隔符，正确读取CSV
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'best_thresholds_double_thresholds.csv'),
        header=0,
        sep=',(?![^()]*\))',  # 只把括号外的逗号当作分隔符
        engine='python'  # 必须指定python引擎以支持正则分隔符
    )
    df.set_index('Table Name', inplace=True)  # 以Table Name为索引

    # 将Best RDC Threshold列的字符串转换为元组
    df['Best RDC Threshold'] = df['Best RDC Threshold'].apply(str_to_tuple)

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
                        # 此时获取的label已经是元组类型
                        label = df.loc[table_name, 'Best RDC Threshold']
                        labels.append(label)
                    else:
                        raise KeyError(f"cant find table name '{table_name}' 对应的阈值")
                except Exception as e:
                    print(f"processing file {matrix_file} failed: {str(e)}")

    print(f"loaded matrix count: {len(data)}, label count: {len(labels)}")
    # 打印第一个标签及其类型，验证是否转换成功
    if labels:
        print(f"value of first label: {labels[0]}, type: {type(labels[0])}")
    return data, labels


if __name__ == "__main__":
    main_folder = "/home/dafn/card/deepcard/collected_datasets"
    data, labels = load_matrices_and_generate_labels_double_threshold(main_folder)
