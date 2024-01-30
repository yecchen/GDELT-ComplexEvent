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

stat_dict = {
    'no_event': 0,
    'invalid_format': 0,
    'listed_triplets': 0,

    'unmatched_relation': 0,
    'unknown_actor': 0,
    'level1_events': 0,  # after check relation and actor
}

filtered_records = {
    'no_event': {}, # md5: rsp
    'invalid_format': {}, # md5: [events]
    'unmatched_relation': {},
    'unknown_actor': {}
}

UNKNOWN_ACTOR = ['unknown', 'none']

@torch.inference_mode()
def main(args, exists_results, dict_id2ont, dict_hier_id):
    # build level 1 relation name2id dict (id is str)
    level1_name2id = {}
    level1_ids = list(dict_hier_id.keys())
    for level1_id in level1_ids:
        level1_name2id[dict_id2ont[level1_id]['choice'].lower()] = level1_id
        level1_name2id[dict_id2ont[level1_id]['name'].lower()] = level1_id
    print('Level 1 relation name2id dict is biult.')

    all_events = []

    for idx, md5 in tqdm(enumerate(exists_results), total=len(exists_results)):

        with open(args.output_path + args.year + '/results/' + md5, 'r') as fraw:
            rsp = fraw.read()
            rsp = rsp.strip(' |\n' + punctuation)

            # check if no structured event is extracted
            if ';' not in rsp:
                stat_dict['no_event'] += 1
                filtered_records['no_event'][md5] = rsp
                continue

            # check each event
            events = rsp.split('|')
            for event in events:
                # check format
                fields = event.split(';')
                if len(fields) != 3:
                    stat_dict['invalid_format'] += 1
                    if md5 not in filtered_records['invalid_format']:
                        filtered_records['invalid_format'][md5] = []
                    filtered_records['invalid_format'][md5].append(event)
                    continue
                s, r, o = [_.strip(' ') for _ in fields]
                stat_dict['listed_triplets'] += 1

                # check relation
                if r.lower() not in level1_name2id:
                    stat_dict['unmatched_relation'] += 1
                    if md5 not in filtered_records['unmatched_relation']:
                        filtered_records['unmatched_relation'][md5] = []
                    filtered_records['unmatched_relation'][md5].append(event)
                    continue
                r_id = level1_name2id[r.lower()]
                r_choice = dict_id2ont[r_id]['choice']

                # check actor
                if s.lower() in UNKNOWN_ACTOR or o.lower() in UNKNOWN_ACTOR:
                    stat_dict['unknown_actor'] += 1
                    if md5 not in filtered_records['unknown_actor']:
                        filtered_records['unknown_actor'][md5] = []
                    filtered_records['unknown_actor'][md5].append(event)
                    continue
                stat_dict['level1_events'] += 1
                all_events.append([s, r_id, r_choice, o, md5])

    all_events_df = pd.DataFrame(all_events,
                        columns=['Subject', 'Relation_id', 'Relation_choice', 'Object', 'Md5'],
                        dtype='string')

    print(stat_dict)

    if not os.path.isdir('./results{}/level1'.format(args.alias)):
        print('Making new dir: ' + './results{}/level1'.format(args.alias))
        os.makedirs('./results{}/level1'.format(args.alias))

    all_events_df.to_csv(path_or_buf = './results{}/level1/'.format(args.alias) + args.year + '_events.csv',
                         sep='\t', index=False)
    json.dump(stat_dict, open('./results{}/level1/'.format(args.alias) + args.year + '_stats.json', 'w'), indent=4)
    json.dump(filtered_records, open('./results{}/level1/'.format(args.alias) + args.year + '_filtered.json', 'w'), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)

    parser.add_argument('--year', type=str, default='2022',
                        help='specify the year of data to extract')
    parser.add_argument('--input_path', type=str, default='./data_cleaned/',
                        help='input data directory')
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

    dict_year2md5 = json.load(open(args.input_path + 'dict_year2md5s.json'))
    md5_list = dict_year2md5[year]
    print('\nNum of docs in year ' + year + ':' + str(len(md5_list)))

    # check results
    exists_results = os.listdir(args.output_path + year + '/results')
    print('\nNum of exists results in year ' + year + ':' + str(len(exists_results)))


    dict_id2ont = json.load(open('./data/dict_id2ont_choice.json', 'r'))
    dict_hier_id = json.load(open('./data/dict_hier_id.json', 'r'))

    main(args, exists_results, dict_id2ont, dict_hier_id)
