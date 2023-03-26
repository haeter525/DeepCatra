import torch
import torch.nn as nn
import numpy as np
import os
import sys
from DeepCatra.learning.lstm_preprocess import encoding
from collections import defaultdict
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)
from DeepCatra.learning.hybrid_model import Hybrid_Network
from DeepCatra.learning.data_reader import get_data
from DeepCatra.learning.model_train import (
    get_split_dataset,
    print_matrix,
    build_model,
)

opcode_dict = encoding()


def split_opcode_seq(opcode_seq, split_n):
    opcode_seq_list = []
    num = int(len(opcode_seq) / split_n)
    diff_len = (num + 1) * split_n - len(opcode_seq)
    arr = np.array(opcode_seq)
    opcode_seq = np.pad(arr, (0, diff_len), "constant")
    for i in range(num + 1):
        opcode_seq_list.append(opcode_seq[i * split_n : split_n * (i + 1)])
    return opcode_seq_list


def preprocess(graph_vertix, graph_edge):
    for i in range(len(graph_edge)):
        graph_edge[i] = np.unique(graph_edge[i], axis=0)

    Edge_list = []

    for i in range(len(graph_edge)):
        edge_list = defaultdict(list)
        for j in range(graph_edge[i].shape[0]):
            # 反向边
            if graph_edge[i][j][1] in edge_list:
                edge_list[graph_edge[i][j][1]].append(
                    (graph_edge[i][j][0] + 5, graph_edge[i][j][2])
                )
            else:
                edge_list[graph_edge[i][j][1]] = [
                    (graph_edge[i][j][0] + 5, graph_edge[i][j][2])
                ]
            # 前向边
            if graph_edge[i][j][2] in edge_list:
                edge_list[graph_edge[i][j][2]].append(
                    (graph_edge[i][j][0], graph_edge[i][j][1])
                )
            else:
                # degree_list.append(n2)
                edge_list[graph_edge[i][j][2]] = [
                    (graph_edge[i][j][0], graph_edge[i][j][1])
                ]

        Edge_list.append(edge_list)

    node_source_list = []
    node_dest_list = []
    edge_type_index_list = []
    dg_list = []

    for x in range(len(graph_edge)):
        node_source = []
        node_dest = []
        edge_type_index = []
        for i in list(sorted(Edge_list[x].keys())):
            for j in list(Edge_list[x][i]):
                node_source.append(j[1])
                edge_type_index.append(j[0])
                node_dest.append(i)

        node_source_list.append(np.int16(node_source))
        node_dest_list.append(np.int16(node_dest))
        edge_type_index_list.append(np.int8(edge_type_index))
    # 生成度向量
    for i in range(len(graph_edge)):
        _, x_unique = np.unique(node_dest_list[i], return_counts=True)

        node_dest_decrease = np.array([x - 1 for x in node_dest_list[i]])
        dg_list.append(np.array(x_unique[node_dest_decrease]))

    return (
        graph_vertix,
        node_source_list,
        node_dest_list,
        edge_type_index_list,
        dg_list,
    )


def test(
    test, model_params_path="model_best_params.pkl", task_type: str = "predict"
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T = 10
    model = Hybrid_Network(13, 32, T)
    model.load_state_dict(torch.load(model_params_path))
    model.to(device)
    model.eval()
    prob_labels = []

    Graph_vertix = test[0]
    Node_source_list = test[1]
    Node_dest_list = test[2]
    Edge_type_index_list = test[3]
    Dg_list = test[4]
    Lstm_feature = test[5]
    labels = test[6]
    with torch.no_grad():
        test_pred = build_model(
            device,
            model,
            prob_labels,
            Graph_vertix,
            Node_source_list,
            Node_dest_list,
            Edge_type_index_list,
            Dg_list,
            Lstm_feature,
        )

        if task_type == "test":
            print_matrix(test, test_pred)
        elif task_type == "predict":
            print("The pred label is ：", test_pred)
            print(
                "The predicted probability for postive calss is", prob_labels
            )


def main():
    test_apk_path = sys.argv[1]
    test_dataset = get_split_dataset(test_apk_path, 13, 100)
    test(test_dataset)


if __name__ == "__main__":
    main()
