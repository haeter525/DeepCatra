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
from DeepCatra.learning.lstm_model import LSTM_net


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


def load_my_data_split(deal_folder, split_length):
    opcode_dict = encoding()
    feature_data = []
    with open(deal_folder, "r", encoding="utf-8") as file:
        opcode_seq = []
        for line in file.readlines():
            line = line.strip("\n")
            if line == "" and len(opcode_seq) != 0:
                feature_data.extend(split_opcode_seq(opcode_seq, split_length))
                opcode_seq = []
            elif line.find(":") == -1 and line != "":
                opcode_seq.append(np.int32(opcode_dict[line]))

        if len(opcode_seq) != 0:
            if len(opcode_seq) >= split_length:
                feature_data.extend(split_opcode_seq(opcode_seq, split_length))
    return feature_data


def get_data(path, ln, split_length):
    graph_edge = []
    graph_vertix = []
    lstm_feature = []

    dir = os.listdir(path)
    for apk in dir:
        apk_path = os.path.join(path, apk)
        vandeando = os.listdir(apk_path)
        edge_path = os.path.join(apk_path, "edge.txt")
        vertix_path = os.path.join(apk_path, "vertix.txt")
        opcode_path = os.path.join(apk_path, "sensitive_opcode_seq.txt")
        for vande in vandeando:
            if vande == "edge.txt":
                edge_info = open(edge_path)
                lines = edge_info.readlines()
                edge = np.zeros((len(lines), 3), dtype=int)
                j = 0
                for line in lines:
                    curline = line.strip("\n")
                    curline = curline.split()
                    curline = [int(i) for i in curline]
                    curline = np.array(curline)
                    edge[j] = curline
                    j += 1
                graph_edge.append(np.array(edge))

            if vande == "vertix.txt":
                vertix_info = open(vertix_path)
                i = 0
                lines = vertix_info.readlines()
                vertix = np.zeros((len(lines), ln), dtype=float)

                for line in lines:
                    curline = line.strip("\n")
                    curline = curline.split()
                    curline = [int(i) for i in curline]

                    if len(curline) < ln:
                        curline = list(curline + [0] * (ln - len(curline)))
                    if len(curline) > ln:
                        curline = curline[:ln]
                    curline = np.array(curline)
                    curline = curline.astype(float)

                    # 归一化
                    curline = curline / 232
                    vertix[i] = curline
                    i += 1
                graph_vertix.append(vertix)

            if vande == "sensitive_opcode_seq.txt":
                single_apk_data = load_my_data_split(opcode_path, split_length)
                single_apk_data = np.array(single_apk_data)
                lstm_feature.append(np.array(single_apk_data))

    return graph_vertix, graph_edge, lstm_feature


def get_split_dataset(path, ln, split_length):
    labels, graph_vertix, graph_edge = get_data(path, ln, split_length)
    (
        graph_vertix,
        node_source_list,
        node_dest_list,
        edge_type_index_list,
        dg_list,
    ) = preprocess(graph_vertix, graph_edge)

    np.random.seed(0)
    indices = np.random.permutation(len(graph_vertix))

    graph_vertix = np.array(graph_vertix, dtype=object)[indices]
    node_source_list = np.array(node_source_list, dtype=object)[indices]
    node_dest_list = np.array(node_dest_list, dtype=object)[indices]
    edge_type_index_list = np.array(edge_type_index_list, dtype=object)[
        indices
    ]
    dg_list = np.array(dg_list, dtype=object)[indices]
    lstm_feature = np.array(lstm_feature, dtype=object)[indices]

    labels = np.array(labels)[indices]
    dataset = [
        graph_vertix,
        node_source_list,
        node_dest_list,
        edge_type_index_list,
        dg_list,
        lstm_feature,
        labels,
    ]
    return dataset


def test(test, task_type: str = "predict"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T = 10
    model = Hybrid_Network(13, 32, T)

    model.load_state_dict(torch.load("model_best_params.pkl"))
    model.to(device)
    model.eval()
    test_pred = []
    prob_labels = []
    Graph_vertix = test[0]

    Node_source_list = test[1]
    Node_dest_list = test[2]
    Edge_type_index_list = test[3]
    Dg_list = test[4]
    Lstm_feature = test[5]
    labels = test[6]
    with torch.no_grad():
        for i in range(len(Graph_vertix)):
            lstm_feature = Lstm_feature[i].astype(int)
            graph_vertix = Graph_vertix[i].astype(float)
            node_source_list = Node_source_list[i].astype(int)
            node_dest_list = Node_dest_list[i].astype(int)
            edge_type_index_list = Edge_type_index_list[i].astype(int)
            dg_list = Dg_list[i].astype(int)

            lstm_feature = torch.LongTensor(lstm_feature)
            graph_vertix = torch.FloatTensor(graph_vertix)
            node_source_list = torch.LongTensor(node_source_list)
            node_dest_list = torch.LongTensor(node_dest_list)
            edge_type_index_list = torch.LongTensor(edge_type_index_list)
            dg_list = torch.LongTensor(dg_list)

            lstm_feature = lstm_feature.to(device)
            graph_vertix = graph_vertix.to(device)
            node_source_list = node_source_list.to(device)
            node_dest_list = node_dest_list.to(device)
            edge_type_index_list = edge_type_index_list.to(device)
            dg_list = dg_list.to(device)

            out = model(
                graph_vertix,
                node_source_list,
                node_dest_list,
                edge_type_index_list,
                dg_list,
                lstm_feature,
            )
            pred = torch.max(out, 1)[1].cpu().numpy()
            prob_label = out.cpu().numpy()
            prob_labels.append(prob_label[0][0])
            test_pred.append(pred[0])

        if task_type == "test":
            test_pred = np.array(test_pred)
            accuracy = accuracy_score(test[6], test_pred)
            precision = precision_score(
                test[6], test_pred, average="binary"
            )  # 输出精度
            recall = recall_score(
                test[6], test_pred, average="binary"
            )  # 输出召回率
            f1 = f1_score(test[6], test_pred, average="binary")
            auc = roc_auc_score(test[6], prob_labels)
            print("accuracy: ", accuracy)
            print("precision: ", precision)
            print("recall: ", recall)
            print("f1-score: ", f1)
            print("auc: ", auc)
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
