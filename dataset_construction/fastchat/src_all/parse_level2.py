"""
Usage:
python3 -m fastchat.serve.huggingface_api --model ~/model_weights/vicuna-7b/
"""
import argparse
import json

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.conversation import get_default_conv_template
from fastchat.serve.inference import load_model, add_model_args

from tqdm import tqdm
import os
from string import punctuation
import re

stat_dict = {
    'not_specified': 0,
    'same_with_level1': 0,
    'unmatched_relation': 0,
    'level2_events': 0,
}

filtered_records = {
    'not_specified': {}, # rowid:response
    'same_with_level1': {},
    'unmatched_relation': {}
}

MANUAL = {
    "Engage in economic cooperation": "Cooperate economically",
    "Engage in military cooperation": "Cooperate militarily",
    "Express optimistic comment": "Make optimistic comment",
    "Cooperate judicially": "Engage in judicial cooperation",
    "Demand for material cooperation": "Demand material cooperation",
    "Demand for material aid": "Demand material aid",
    "Demand for political reform": "Demand political reform",
    "Demand for meeting, negotiation": "Demand meeting, negotiation",
    "Demand for settling of dispute": "Demand settling of dispute",
    "Demand for mediation": "Demand mediation",
    "Investigate corruption": "Investigate crime, corruption",
    "Investigate crime": "Investigate crime, corruption",
    "Express intent to engage in political reform": "Express intent to institute political reform",
    "Fight with aerial weapons": "Employ aerial weapons",
    "Demand for a cease-fire": "Demand that target yield or concede",
    "Express intent to engage in military cooperation": "Express intent to engage in material cooperation",
    "Appeal for military cooperation": "Appeal for material cooperation",
    "Appeal for judicial cooperation": "Appeal for material cooperation",
}

MANUAL_PATTERNS = {
    "Demand .* release .*": "Demand that target yield or concede"
}

@torch.inference_mode()
def main(args, level1_event_df, exists_results, dict_id2ont, dict_hier_id):
    # build level 2 relation name2id dict (id is str)
    level2_name2id = {}
    level2_ids = []
    for level1, info in dict_hier_id.items():
        level2_ids += list(info.keys())
    for level2_id in level2_ids:
        level2_name2id[dict_id2ont[level2_id]['choice'].lower()] = level2_id
    print('Level 2 relation name2id dict is biult.')

    all_events = []

    for idx, rowid in tqdm(enumerate(exists_results), total=len(exists_results)):

        level1_row = level1_event_df.iloc[[rowid]]
        s = level1_row['Subject'].item()
        level1_r_choice = level1_row['Relation_choice'].item()
        o = level1_row['Object'].item()
        md5 = level1_row['Md5'].item()

        with open(args.output_path + args.year + '/results_level2/' + rowid, 'r') as fraw:
            rsp = fraw.read()
            rsp = rsp.strip(' |\n' + punctuation)

            # allow some flexible expressions
            if '.' in rsp or ':' in rsp:
                sent = rsp.split('.')[0]
                span_start = sent.find(' is')
                if span_start != -1:
                    rsp = sent[span_start + 3:].strip(punctuation + ' \"\':')

            # check not specified
            if rsp.lower() == 'not specified':
                stat_dict['not_specified'] += 1
                filtered_records['not_specified'][rowid] = rsp
                continue

            # check if is the same with level 1 relation
            if rsp.lower() == level1_r_choice.lower():
                stat_dict['same_with_level1'] += 1
                filtered_records['same_with_level1'][rowid] = rsp
                continue

            # check unmatched relation
            if rsp.lower() not in level2_name2id:
                if rsp in MANUAL:
                    rsp = MANUAL[rsp]
                else:
                    for pattern in MANUAL_PATTERNS:
                        if re.match(pattern, rsp) is not None:
                            rsp = MANUAL_PATTERNS[pattern]
                            break

            if rsp.lower() not in level2_name2id:
                stat_dict['unmatched_relation'] += 1
                filtered_records['unmatched_relation'][rowid] = rsp
                continue

            level2_r_id = level2_name2id[rsp.lower()]
            level2_r_choice = dict_id2ont[level2_r_id]['choice']

            stat_dict['level2_events'] += 1
            all_events.append([rowid, s, level2_r_id, level2_r_choice, o, md5])

    all_events_df = pd.DataFrame(all_events,
                        columns=['Level1_rowid', 'Subject', 'Relation_id', 'Relation_choice', 'Object', 'Md5'],
                        dtype='string')

    print(stat_dict)

    if not os.path.isdir('./results{}/level2'.format(args.alias)):
        print('Making new dir: ' + './results{}/level2'.format(args.alias))
        os.makedirs('./results{}/level2'.format(args.alias))

    all_events_df.to_csv(path_or_buf = './results{}/level2/'.format(args.alias) + args.year + '_events.csv',
                         sep='\t', index=False)
    json.dump(stat_dict, open('./results{}/level2/'.format(args.alias) + args.year + '_stats.json', 'w'), indent=4)
    json.dump(filtered_records, open('./results{}/level2/'.format(args.alias) + args.year + '_filtered.json', 'w'), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)

    parser.add_argument('--year', type=str, default='2022',
                        help='specify the year of data to extract')
    parser.add_argument('--output_path', type=str, default='/ssd1/ccye/vicuna-event/',
                        help='output data directory')
    parser.add_argument('--alias', type=str, default='',
                        help='parsing results data directory, e.g. _remain')

    args = parser.parse_args()

    # specify the year of data
    year = args.year

    if not os.path.isdir(args.output_path + year):
        print('Invalid dir: ' + args.output_path + year)
        exit()

    # load level 1 events
    level1_event_df = pd.read_csv('./results{}/level1/'.format(args.alias) + year + '_events.csv', sep='\t', dtype='string')
    print('\nNum of level 1 events in year ' + year + ':' + str(len(level1_event_df)))

    # check results
    exists_results = os.listdir(args.output_path + year + '/results_level2')
    print('\nNum of exists level 2 responses in year ' + year + ':' + str(len(exists_results)))


    dict_id2ont = json.load(open('./data/dict_id2ont_choice.json', 'r'))
    dict_hier_id = json.load(open('./data/dict_hier_id.json', 'r'))

    main(args, level1_event_df, exists_results, dict_id2ont, dict_hier_id)
