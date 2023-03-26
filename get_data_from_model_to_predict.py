def get_data(path, ln,split_length):
    graph_edge = []
    graph_vertix = []
    lstm_feature = []

    dir = os.listdir(path)
    for apk in dir:
        apk_path = os.path.join(path,apk)
        vandeando = os.listdir(apk_path)
        edge_path = os.path.join(apk_path, 'edge.txt')
        vertix_path = os.path.join(apk_path, 'vertix.txt')
        opcode_path = os.path.join(apk_path, 'sensitive_opcode_seq.txt')
        for vande in vandeando:
            if (vande == 'edge.txt'):
                edge_info = open(edge_path)
                lines = edge_info.readlines()
                edge = np.zeros((len(lines), 3), dtype=int)
                j = 0
                for line in lines:
                    curline = line.strip('\n')
                    curline = curline.split()
                    curline = [int(i) for i in curline]
                    curline = np.array(curline)
                    edge[j] = curline
                    j += 1
                graph_edge.append(np.array(edge))

            if (vande == 'vertix.txt'):
                vertix_info = open(vertix_path)
                i = 0
                lines = vertix_info.readlines()
                vertix = np.zeros((len(lines), ln), dtype=float)

                for line in lines:
                    curline = line.strip('\n')
                    curline = curline.split()
                    curline = [int(i) for i in curline]

                    if (len(curline) < ln):
                        curline = list(curline + [0] * (ln - len(curline)))
                    if (len(curline) > ln):
                        curline = curline[:ln]
                    curline = np.array(curline)
                    curline = curline.astype(float)

                    # 归一化
                    curline = curline / 232
                    vertix[i] = curline
                    i += 1
                graph_vertix.append(vertix)

            if (vande == 'sensitive_opcode_seq.txt'):
                single_apk_data = load_my_data_split(opcode_path, split_length)
                single_apk_data = np.array(single_apk_data)
                lstm_feature.append(np.array(single_apk_data))

    return graph_vertix, graph_edge, lstm_feature