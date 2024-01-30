import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import logging
import os


class RGCNLinkDataset(object):
    def __init__(self, name, dir=None):
        self.name = name
        if dir:
            self.dir = dir
            self.dir = os.path.join(self.dir, self.name)
        print(self.dir)


    def load(self):
        stat_path = os.path.join(self.dir,  'stat.txt')
        entity_path = os.path.join(self.dir, 'entity2id.txt')
        relation_path = os.path.join(self.dir, 'relation2id.txt')
        train_path = os.path.join(self.dir, 'train_query.txt')
        valid_path = os.path.join(self.dir, 'valid_query.txt')
        test_path = os.path.join(self.dir, 'test_query.txt')
        train_tkg_path = os.path.join(self.dir, 'train.txt')
        valid_tkg_path = os.path.join(self.dir, 'valid.txt')
        test_tkg_path = os.path.join(self.dir, 'test.txt')
        outlier_path = os.path.join(self.dir, 'outliers.txt')
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        self.train = np.array(_read_quintuplets_as_list(train_path))
        self.valid = np.array(_read_quintuplets_as_list(valid_path))
        self.test = np.array(_read_quintuplets_as_list(test_path))
        self.train_tkg = np.array(_read_quintuplets_as_list(train_tkg_path))
        self.valid_tkg = np.array(_read_quintuplets_as_list(valid_tkg_path))
        self.test_tkg = np.array(_read_quintuplets_as_list(test_tkg_path))
        self.outlier = np.array(_read_quintuplets_as_list(outlier_path))
        with open(os.path.join(self.dir, 'stat.txt'), 'r') as f:
            line = f.readline()
            num_nodes, num_rels = line.strip().split("\t")
            num_nodes = int(num_nodes)
            num_rels = int(num_rels)
        self.num_nodes = num_nodes
        self.num_rels = len(relation_dict)
        self.relation_dict = relation_dict
        self.entity_dict = entity_dict
        print("# Sanity Check:  entities: {}".format(self.num_nodes))
        print("# Sanity Check:  relations: {}".format(self.num_rels))
        print("# Sanity Check:  edges: {}".format(len(self.train)))


def _read_dictionary(filename):
    d = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            d[int(line[1])] = line[0]
    return d


def _read_quintuplets(filename):
    with open(filename, 'r') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line


def _read_quintuplets_as_list(filename):
    l = []
    for triplet in _read_quintuplets(filename):
        s = int(triplet[0])
        r = int(triplet[1])
        o = int(triplet[2])
        t = int(triplet[3])
        ceid = int(triplet[4])
        l.append([s, r, o, t, ceid])
    return l


def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #

    return score


def get_total_rank(test_triples, score, all_ans, eval_bz):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz

    filter_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]

        target = test_triples[batch_start:batch_end, 2]

        filter_score_batch = filter_score(triples_batch, score_batch, all_ans)
        filter_rank.append(sort_and_rank(filter_score_batch, target))

    filter_rank = torch.cat(filter_rank)
    filter_rank += 1
    filter_mrr = torch.mean(1.0 / filter_rank.float())

    return filter_mrr.item(), filter_rank


def get_filtered_score(test_triples, score, all_ans, eval_bz):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz

    filtered_score = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        filtered_score.append(filter_score(triples_batch, score_batch, all_ans))
    filtered_score = torch.cat(filtered_score,dim =0)

    return filtered_score


def popularity_map(tuple_tensor, head_ents):
    tags = 'head' if tuple_tensor[2].item() in head_ents else 'other'
    return tags


def cal_ranks(rank_list, tags_all, mode):
    total_tag_all = []
    hits = [1, 3, 10]
    rank_list = torch.cat(rank_list)
    for tag_all in tags_all:
        total_tag_all += tag_all

    all_df = pd.DataFrame({'rank_ent': rank_list.cpu(), 'ent_tag': total_tag_all})
    debiased_df = all_df[all_df['ent_tag'] != 'head']
    debiased_rank_ent = torch.tensor(list(debiased_df['rank_ent']))
    mrr_debiased = torch.mean(1.0 / debiased_rank_ent.float())

    if mode == 'test':
        logging.info("====== object prediction ======")
        logging.info("MRR: {:.6f}".format(mrr_debiased.item()))
        for hit in hits:
            avg_count_ent_debiased = torch.mean((debiased_rank_ent <= hit).float())
            logging.info("Hits@ {}: {:.6f}".format(hit, avg_count_ent_debiased.item()))

    return mrr_debiased


def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    def add_subject(e1, e2, r, d, num_rel):
        if not e2 in d:
            d[e2] = {}
        if not r + num_rel in d[e2]:
            d[e2][r + num_rel] = set()
        d[e2][r + num_rel].add(e1)

    def add_object(e1, e2, r, d, num_rel):
        if not e1 in d:
            d[e1] = {}
        if not r in d[e1]:
            d[e1][r] = set()
        d[e1][r].add(e2)

    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_dict = {}
    all_snap = list(split_by_time(total_data).values())
    all_times = np.array(sorted(set(total_data[:, 3]))) 
    for time, snap in zip(all_times, all_snap):
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_dict[time] = all_ans_t

    return all_ans_dict

def map_time2query_ceids(total_data):
    time2query_ceids = {}
    for line in total_data:
        t, ceid = line[3:]
        if t not in time2query_ceids:
            time2query_ceids[t] = set()
        time2query_ceids[t].add(ceid)
    # sort ceid order
    for t, ceidset in time2query_ceids.items():
        time2query_ceids[t] = sorted(list(ceidset))
    return time2query_ceids

def split_by_time(arr):
    time_dict = dict()

    for row in arr:
        time = row[3]  # Get the time value
        if time not in time_dict:
            time_dict[time] = []
        time_dict[time].append(row[:3])

    # Convert lists of rows back into arrays
    for time in time_dict:
        time_dict[time] = np.array(time_dict[time])

    snapshot_list = list(time_dict.values())

    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list])))

    return time_dict

def split_by_time_ceid(arr, num_rels):
    # add reverse query here
    time_ceid_dict = dict() # {t: {ceid: [] } }
    time_dict = dict() # {t: []}

    for row in arr:
        time = row[3]  # Get the time value
        ceid = row[4]
        if time not in time_ceid_dict:
            time_ceid_dict[time] = dict()
        if ceid not in time_ceid_dict[time]:
            time_ceid_dict[time][ceid] = []
        time_ceid_dict[time][ceid].append(row[:3])

    # Convert lists of rows back into arrays and add reverse query
    for time, info in time_ceid_dict.items():
        # sorted ceid
        ceids = sorted(list(info.keys()))
        t_queries = []
        for ceid in ceids:
            queries = np.array(info[ceid])
            rev_queries = queries[:, [2, 1, 0]]
            rev_queries[:, 1] = rev_queries[:, 1] + num_rels
            all_queries = np.concatenate([queries, rev_queries])
            time_ceid_dict[time][ceid] = all_queries
            t_queries.append(all_queries)
        time_dict[time] = np.concatenate(t_queries)

    return time_dict, time_ceid_dict

