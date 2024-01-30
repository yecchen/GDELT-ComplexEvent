import os
import pickle
import yaml
import argparse
import logging
import time
import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
import json
from tqdm import tqdm
import random
import utils
from models.logo_sep import LoGo_sep
from models.logo_share import LoGo_share


def get_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--gpu", type=str, default=0, help="which gpu to use")
    parser.add_argument("-d", "--dataset", type=str, default="MIDEAST_CE", help="MIDEAST_CE, GDELT_CE")
    parser.add_argument("-m", "--model", type=str, default="LoGo_share", help="which model to use, options: LoGo_share, LoGo_sep")
    parser.add_argument("-i", "--info", type=str, default="late", help="additional info for certain run")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--local_hist_len", type=int, default=5, help="local history length")
    parser.add_argument("--global_hist_len", type=int, default=5, help="global history length")
    parser.add_argument("--n_layers", type=int, default=2, help="graph propagation layers")
    parser.add_argument("--local_only", action='store_true', help="use local graph only")
    parser.add_argument("--global_only", action='store_true', help="use global graph only")
    args = parser.parse_args()

    return args


def test(conf, model, model_name,
         test_times, test_list, test_list_with_ceid,
         global_graph_dict, local_graph_dict,
         global_times, time2query_ceids, ceid2local_times,
         all_ans_dict, head_ents, mode='eval'):
    if mode == "test":
        # test mode: load parameter form file
        checkpoint = torch.load(model_name, map_location=conf["device"])
        logging.info("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        logging.info("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    device = conf["device"]
    global_hist_len = conf["global_hist_len"]
    local_hist_len = conf["local_hist_len"]
    local_only = conf["local_only"]
    global_only = conf["global_only"]

    ranks_filter, mrr_filter_list = [], []

    tags, tags_all = [], []

    model.eval()

    query_times = list(test_times)

    for batch_idx, query_time in enumerate(tqdm(query_times)):
        global_triplets = torch.LongTensor(test_list[query_time]).to(device)
        if not local_only:
            # global input
            if query_time - global_hist_len < 0:
                global_hist_list = global_times[0: query_time]
            else:
                global_hist_list = global_times[query_time - global_hist_len: query_time]
            global_g_list = [global_graph_dict[tim].to(device) for tim in global_hist_list]
            # global forward
            global_scores, global_query_embs, global_ent_embs = model.predict_global(global_g_list, global_triplets)
            if global_only:
                final_score = model.predict_query(global_query_embs, global_ent_embs)
        if not global_only:
            # local: every ceid at t
            final_score = []
            query_idx = 0
            for ceid in time2query_ceids[query_time]:
                # local input
                local_times = ceid2local_times[ceid]
                query_time_local_idx = local_times.index(query_time)
                if query_time_local_idx - local_hist_len < 0:
                    local_hist_list = local_times[0:query_time_local_idx]
                else:
                    local_hist_list = local_times[query_time_local_idx - local_hist_len: query_time_local_idx]
                local_g_list = [local_graph_dict[ceid][tim].to(device) for tim in local_hist_list]
                local_triplets = torch.LongTensor(test_list_with_ceid[query_time][ceid]).to(device)
                # local forward
                local_scores, local_query_embs, local_ent_embs = model.predict_local(local_g_list, local_triplets)
                if local_only:
                    final_score.append(model.predict_query(local_query_embs, local_ent_embs))
                else:
                    # late fusion for query and obj embedding
                    corr_global_query_embs = global_query_embs[query_idx: query_idx + len(local_triplets), :]
                    final_score.append(model.predict_query(corr_global_query_embs + local_query_embs, global_ent_embs + local_ent_embs))
                query_idx += len(local_triplets)
            final_score = torch.cat(final_score)

        mrr_filter, rank_filter = utils.get_total_rank(global_triplets, final_score, all_ans_dict[query_time], eval_bz=1000)

        popularity_tag = list(map(lambda x: utils.popularity_map(x, head_ents), global_triplets))
        tags_all.append(popularity_tag)

        ranks_filter.append(rank_filter)
        mrr_filter_list.append(mrr_filter)

    mrr_filter_all = utils.cal_ranks(ranks_filter, tags_all, mode)

    return mrr_filter_all


def main():
    conf = yaml.safe_load(open("./config.yaml"))
    print("load config file done!")

    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]

    conf = conf[dataset_name]
    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]
    conf["model"] = paras["model"]
    conf["dataset"] = dataset_name
    conf["data_path"] = conf["path"] + "/" + conf["dataset"]
    conf["lr"] = paras['lr']
    conf["wd"] = paras['wd']
    conf['local_hist_len'] = paras['local_hist_len']
    conf['global_hist_len'] = paras['global_hist_len']
    conf['n_layers'] = paras['n_layers']
    conf['local_only'] = paras['local_only']
    conf['global_only'] = paras['global_only']
    if conf['local_only']:
        print('Local Only!')
        conf["info"] = conf["info"] + '-Local' if conf["info"] != "" else 'Local'
    if conf['global_only']:
        print('Global Only!')
        conf["info"] = conf["info"] + '-Global' if conf["info"] != "" else 'Global'

    device = torch.device(f"cuda:{conf['gpu']}" if torch.cuda.is_available() else "cpu")
    conf["device"] = device

    # load data
    print("loading popularity bias data")
    head_ents = json.load(open('{}/head_ents.json'.format(conf["data_path"]), 'r'))

    print("loading training graphs...")
    with open(os.path.join(conf["data_path"], 'global_graph_dict.pkl'), 'rb') as fp:
        global_graph_dict = pickle.load(fp) # {t: g}
    with open(os.path.join(conf["data_path"], 'local_graph_dict.pkl'), 'rb') as fp:
        local_graph_dict = pickle.load(fp) # {ceid: {t: g}}, t is sorted

    ceid2local_times = {}
    for ceid, info in local_graph_dict.items():
        local_times = list(info.keys())
        ceid2local_times[ceid] = local_times

    data = utils.RGCNLinkDataset(conf["dataset"], conf["path"])
    data.load()

    num_ents = data.num_nodes
    num_rels = data.num_rels
    conf["num_ents"] = num_ents
    conf["num_rels"] = num_rels

    train_times = np.array(sorted(set(data.train[:, 3])))
    val_times = np.array(sorted(set(data.valid[:, 3])))
    test_times = np.array(sorted(set(data.test[:, 3])))
    # query quadruplets (s, r, o, ceid)
    # dict of t to dict of int to np array: {t: {ceid: [(s,r,o)...at t]} for  all t appeared in all queries}
    # add reverse query here
    train_list, train_list_with_ceid = utils.split_by_time_ceid(data.train, num_rels)
    valid_list, valid_list_with_ceid = utils.split_by_time_ceid(data.valid, num_rels)
    test_list, test_list_with_ceid = utils.split_by_time_ceid(data.test, num_rels)

    all_data = np.concatenate([data.train_tkg, data.valid_tkg, data.test_tkg, data.outlier])
    global_times = sorted(np.unique(all_data[:, 3]))

    all_query_data = np.concatenate([data.train, data.valid, data.test])
    all_ans_dict = utils.load_all_answers_for_time_filter(all_query_data, num_rels, num_ents, False)
    time2query_ceids = utils.map_time2query_ceids(all_query_data)

    # initialize log
    model_name = "{}-{}-lr{}-wd{}-hisl{}-hisg{}-n{}".format(conf["model"], conf["info"], conf["lr"], conf["wd"], conf['local_hist_len'], conf['global_hist_len'], conf['n_layers'])
    model_path = './checkpoints/{}/'.format(conf["dataset"])
    model_state_file = model_path + model_name
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    log_path = './logs/{}/'.format(conf["dataset"])
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(level=logging.INFO, filename=log_path + model_name + '.log')

    run_path = "./runs/{}/{}".format(conf["dataset"], model_name)
    if not os.path.isdir(run_path):
        os.makedirs(run_path)
    logging.info("Sanity Check: stat name : {}".format(model_state_file))
    run_path = "./runs/{}/{}".format(conf["dataset"], model_name)
    if not os.path.isdir(run_path):
        os.makedirs(run_path)

    run = SummaryWriter(run_path)

    # build model
    if conf['model'] == 'LoGo_share':
        model = LoGo_share(conf)
        print('Model: LoGo_share_late')
    elif conf['model'] == 'LoGo_sep':
        model = LoGo_sep(conf)
        print('Model: LoGo_sep_late')
    else:
        raise Exception('Unknown model!')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=conf["wd"])

    # start training
    print("-----------------------------start training-------------------------------n")
    global_hist_len = conf["global_hist_len"]
    local_hist_len = conf["local_hist_len"]
    local_only = conf["local_only"]
    global_only = conf["global_only"]

    best_val_mrr, best_test_mrr = 0, 0
    accumulated = 0
    epoch_times = []
    for epoch in range(conf["n_epochs"]):
        epoch_start_time = time.time()
        model.train()

        query_times = list(train_times)
        random.shuffle(query_times)

        losses = []
        epoch_anchor = epoch * len(query_times)
        for batch_idx, query_time in enumerate(tqdm(query_times)):
            global_triplets = torch.LongTensor(train_list[query_time]).to(device)
            if not local_only:
                # global input
                if query_time - global_hist_len < 0:
                    global_hist_list = global_times[0: query_time]
                else:
                    global_hist_list = global_times[query_time - global_hist_len: query_time]
                global_g_list = [global_graph_dict[tim].to(device) for tim in global_hist_list]
                # global forward
                global_scores, global_query_embs, global_ent_embs = model.predict_global(global_g_list, global_triplets)
                if global_only:
                    loss = model.forward_query(global_query_embs, global_ent_embs, global_triplets) * len(global_triplets)
            if not global_only:
                # local: every ceid at t
                loss = torch.zeros(1).to(device)
                query_idx = 0
                for ceid in time2query_ceids[query_time]:
                    # local input
                    local_times = ceid2local_times[ceid]
                    query_time_local_idx = local_times.index(query_time)
                    if query_time_local_idx - local_hist_len < 0:
                        local_hist_list = local_times[0:query_time_local_idx]
                    else:
                        local_hist_list = local_times[query_time_local_idx-local_hist_len: query_time_local_idx]
                    local_g_list = [local_graph_dict[ceid][tim].to(device) for tim in local_hist_list]
                    local_triplets = torch.LongTensor(train_list_with_ceid[query_time][ceid]).to(device)
                    # local forward
                    local_scores, local_query_embs, local_ent_embs = model.predict_local(local_g_list, local_triplets)
                    if local_only:
                        loss += model.forward_query(local_query_embs, local_ent_embs, local_triplets) * len(local_triplets)
                    else:
                        # late fusion for query and obj embedding
                        corr_global_query_embs = global_query_embs[query_idx: query_idx + len(local_triplets), :]
                        loss += model.forward_query(corr_global_query_embs + local_query_embs, global_ent_embs + local_ent_embs, local_triplets) * len(local_triplets)
                    query_idx += len(local_triplets)

            if loss == 0:
                continue

            loss = loss / len(global_triplets)
            losses.append(loss.item())

            batch_anchor = epoch_anchor + batch_idx
            run.add_scalar('loss/loss', loss.item(), batch_anchor)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf["grad_norm"])  # clip gradients
            optimizer.step()
            optimizer.zero_grad()

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)

        average_epoch_time = sum(epoch_times) / len(epoch_times)
        total_epoch_time = sum(epoch_times)

        logging.info(f'Epoch {epoch + 1}/{conf["n_epochs"]}, Time: {epoch_time:.2f}s, AvgTime: {average_epoch_time:.2f}s, TotTime: {total_epoch_time:.2f}s, Loss: {np.mean(losses)}')

        print("Epoch {:04d}, AveLoss: {:.4f}, BestMRRVal: {:.4f}, BestMRRTest: {:.4f}, Model: {}, Dataset: {}".format(epoch, np.mean(losses), best_val_mrr, best_test_mrr, conf["model"], conf["dataset"]))

        # validation and test
        if (epoch + 1) % conf["test_interval"] == 0:
            val_mrr = test(conf=conf,
                           model=model,
                           model_name=model_state_file,
                           test_times=val_times,
                           test_list=valid_list,
                           test_list_with_ceid=valid_list_with_ceid,
                           global_graph_dict=global_graph_dict,
                           local_graph_dict=local_graph_dict,
                           global_times=global_times,
                           time2query_ceids=time2query_ceids,
                           ceid2local_times=ceid2local_times,
                           all_ans_dict=all_ans_dict,
                           head_ents=head_ents,
                           mode='eval')
            run.add_scalar('val/mrr', val_mrr, epoch)

            test_mrr = test(conf=conf,
                            model=model,
                            model_name=model_state_file,
                            test_times=test_times,
                            test_list=test_list,
                            test_list_with_ceid=test_list_with_ceid,
                            global_graph_dict=global_graph_dict,
                            local_graph_dict=local_graph_dict,
                            global_times=global_times,
                            time2query_ceids=time2query_ceids,
                            ceid2local_times=ceid2local_times,
                            all_ans_dict=all_ans_dict,
                            head_ents=head_ents,
                            mode='eval')
            run.add_scalar('test/mrr', test_mrr, epoch)

            if val_mrr < best_val_mrr:
                accumulated += 1
                if epoch >= conf["n_epochs"]:
                    print("Max epoch reached! Training done.")
                    break
                if accumulated >= conf["patience"]:
                    print("Early stop triggered! Training done at epoch{}".format(epoch))
                    break
            else:
                accumulated = 0
                best_val_mrr = val_mrr
                best_test_mrr = test_mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

    test(conf=conf,
        model=model,
        model_name=model_state_file,
        test_times=test_times,
        test_list=test_list,
        test_list_with_ceid=test_list_with_ceid,
        global_graph_dict=global_graph_dict,
        local_graph_dict=local_graph_dict,
        global_times=global_times,
        time2query_ceids=time2query_ceids,
        ceid2local_times=ceid2local_times,
        all_ans_dict=all_ans_dict,
        head_ents=head_ents,
        mode='test')


if __name__ == '__main__':
    main()
