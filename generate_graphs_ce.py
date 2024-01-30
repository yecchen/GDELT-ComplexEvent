import numpy as np
import os
import pickle
import dgl
import torch
from tqdm import tqdm
import argparse
from collections import defaultdict


def load_quadruples(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            ceid = int(line_split[4])
            quadrupleList.append([head, rel, tail, time, ceid])
            times.add(time)

    times = list(times)
    times.sort()

    return np.array(quadrupleList), np.asarray(times)


def get_data_with_t(data, tim):
    x = data[np.where(data[:, 3] == tim)].copy()
    x = np.delete(x, [3, 4], 1)  # drops time and ceid column
    return x


def get_data_with_t_ceid(data, tim, ceid):
    x = data[np.where((data[:, 3] == tim) & (data[:, 4] == ceid))].copy()
    x = np.delete(x, [3, 4], 1)  # drops 3rd column
    return x


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    # norm = 1.0 / in_deg
    norm = in_deg ** -0.5
    return norm


def r2e(triplets, num_rels):
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r + num_rels))
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        r_to_e[rel + num_rels].add(src)
        r_to_e[rel + num_rels].add(dst)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx, idx + len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx


def get_big_graph(triples, num_nodes, num_rels):
    src, rel, dst = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)

    uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    g.uniq_r = uniq_r
    g.r_to_e = r_to_e
    g.r_len = r_len

    return g


def main(args):
    global_graph_dict, local_graph_dict = {}, {}

    data_path = "./data/" + args.dataset

    train_data, train_times = load_quadruples(data_path, 'train.txt')
    val_data, val_times = load_quadruples(data_path, 'valid.txt')
    test_data, test_times = load_quadruples(data_path, 'test.txt')
    outlier_data, outlier_times = load_quadruples(data_path, 'outliers.txt')

    all_data = np.concatenate([train_data, val_data, test_data, outlier_data])
    all_time = sorted(np.unique(all_data[:, 3]))
    all_ceid = sorted(np.unique(all_data[:, -1]))
    ceid2times = {} #{ceid: [t1, ...]}
    for line in all_data:
        tim, ceid = line[3:]
        if ceid not in ceid2times:
            ceid2times[ceid] = set()
        ceid2times[ceid].add(tim)

    with open(os.path.join(data_path, 'stat.txt'), 'r') as f:
        line = f.readline()
        num_nodes, num_r = line.strip().split("\t")
        num_nodes = int(num_nodes)
        num_r = int(num_r)
    print(num_nodes, num_r)

    with tqdm(total=len(all_time), desc="Generating graphs for global graph") as pbar:
        for tim in all_time:
            data = get_data_with_t(all_data, tim)
            global_graph_dict[tim] = (get_big_graph(data, num_nodes, num_r))
            pbar.update(1)

    with tqdm(total=len(all_ceid), desc="Generating graphs for local graph") as pbar:
        local_graph_dict = dict() # {ceid: {t: g}}, t is sorted
        for ceid in all_ceid:
            if ceid == -1:
                continue
            local_graph_dict[ceid] = dict()
            local_times = list(ceid2times[ceid])
            local_times.sort()
            for tim in local_times:
                data = get_data_with_t_ceid(all_data, tim, ceid)
                local_graph_dict[ceid][tim] = (get_big_graph(data, num_nodes, num_r))
            pbar.update(1)


    with open(os.path.join(data_path, 'global_graph_dict.pkl'), 'wb') as fp:
        pickle.dump(global_graph_dict, fp)

    with open(os.path.join(data_path, 'local_graph_dict.pkl'), 'wb') as fp:
        pickle.dump(local_graph_dict, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate graphs')
    parser.add_argument("--dataset", type=str, default="MIDEAST_CE",
                        help="dataset to generate graph")
    args = parser.parse_args()

    main(args)
