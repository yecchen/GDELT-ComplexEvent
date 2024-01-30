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
    'same_with_level2': 0,
    'unmatched_relation': 0,
    'level3_events': 0,
}

filtered_records = {
    'not_specified': {}, # rowid:response
    'same_with_level2': {},
    'unmatched_relation': {}
}

MANUAL = {
    "Appeal for financial support": "Appeal for economic cooperation"
}

MANUAL_PATTERNS = {
}

@torch.inference_mode()
def main(args, level2_event_df, exists_results, dict_id2ont, dict_hier_id):
    # build level 2 relation name2id dict (id is str)
    level3_name2id = {}
    level3_ids = []
    for level2, level3s in dict_hier_id.items():
        level3_ids += level3s
    for level3_id in level3_ids:
        level3_name2id[dict_id2ont[level3_id]['choice'].lower()] = level3_id
    print('Level 3 relation name2id dict is biult.')

    all_events = []

    for idx, rowid in tqdm(enumerate(exists_results), total=len(exists_results)):

        level2_row = level2_event_df.iloc[[rowid]]
        level2_rid = level2_row['Relation_choice'].item()
        if level2_rid in dict_hier_id:
            print(rowid)
        s = level2_row['Subject'].item()
        level2_r_choice = level2_row['Relation_choice'].item()
        o = level2_row['Object'].item()
        md5 = level2_row['Md5'].item()

        with open(args.output_path + args.year + '/results_level3/' + rowid, 'r') as fraw:
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
            if rsp.lower() == level2_r_choice.lower():
                stat_dict['same_with_level2'] += 1
                filtered_records['same_with_level2'][rowid] = rsp
                continue

            # check unmatched relation
            if rsp.lower() not in level3_name2id:
                if rsp in MANUAL:
                    rsp = MANUAL[rsp]
                else:
                    for pattern in MANUAL_PATTERNS:
                        if re.match(pattern, rsp) is not None:
                            rsp = MANUAL_PATTERNS[pattern]
                            break

            if rsp.lower() not in level3_name2id:
                stat_dict['unmatched_relation'] += 1
                filtered_records['unmatched_relation'][rowid] = rsp
                continue

            level3_r_id = level3_name2id[rsp.lower()]
            level3_r_choice = dict_id2ont[level3_r_id]['choice']

            stat_dict['level3_events'] += 1
            all_events.append([rowid, s, level3_r_id, level3_r_choice, o, md5])

    all_events_df = pd.DataFrame(all_events,
                        columns=['Level3_rowid', 'Subject', 'Relation_id', 'Relation_choice', 'Object', 'Md5'],
                        dtype='string')

    print(stat_dict)

    if not os.path.isdir('./results{}/level3'.format(args.alias)):
        print('Making new dir: ' + './results{}/level3'.format(args.alias))
        os.makedirs('./results{}/level3'.format(args.alias))

    all_events_df.to_csv(path_or_buf = './results{}/level3/'.format(args.alias) + args.year + '_events.csv',
                         sep='\t', index=False)
    json.dump(stat_dict, open('./results{}/level3/'.format(args.alias) + args.year + '_stats.json', 'w'), indent=4)
    json.dump(filtered_records, open('./results{}/level3/'.format(args.alias) + args.year + '_filtered.json', 'w'), indent=4)


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

    # load level 2 events
    level2_event_df = pd.read_csv('./results{}/level2/'.format(args.alias) + year + '_events.csv', sep='\t', dtype='string')
    print('\nNum of level 2 events in year ' + year + ':' + str(len(level2_event_df)))

    # check results
    exists_results = os.listdir(args.output_path + year + '/results_level3')
    print('\nNum of exists level 3 responses in year ' + year + ':' + str(len(exists_results)))


    dict_id2ont = json.load(open('./data/dict_id2ont_choice.json', 'r'))
    dict_hier_id = json.load(open('./data/dict_hier_id_level23.json', 'r'))

    main(args, level2_event_df, exists_results, dict_id2ont, dict_hier_id)
