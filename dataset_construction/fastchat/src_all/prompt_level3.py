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

from fastchat.src_all.prompt_generator import PromptGenerator


@torch.inference_mode()
def main(args, generator, doc_list, level2_event_df, exists_results, dict_hier_id, start_rowid, end_rowid):

    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        debug=args.debug,
    )

    all_msgs = []
    all_rsps = []

    num_no_level3 = 0

    for idx, row in tqdm(level2_event_df.iterrows(), total=len(level2_event_df)):
        # if idx > 15:
        #     break
        if idx in exists_results:
            continue

        if idx < start_rowid or idx > end_rowid:
            continue

        try:
            # check if level3 sub-relation exists
            level2_id = row['Relation_id']
            if level2_id not in dict_hier_id:
                num_no_level3 += 1
                continue

            md5 = row['Md5']
            article = doc_list[md5]
            msg = generator.get_prompt_3_relation(article, row)

            conv = get_default_conv_template(args.model_path).copy()
            conv.append_message(conv.roles[0], msg)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer([prompt]).input_ids
            output_ids = model.generate(
                torch.as_tensor(input_ids).cuda(),
                do_sample=True,
                temperature=0.001,
                max_new_tokens=args.max_new_tokens,
            )
            if model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(input_ids[0]):]
            rsp = tokenizer.decode(output_ids, skip_special_tokens=True,
                                       spaces_between_special_tokens=False)

            with open(args.output_path + args.year + '/results_level3/' + str(idx), 'w') as fresult:
                fresult.write(rsp)

            # all_msgs.append(msg)
            # all_rsps.append(rsp)

        except Exception as e:
            print('num of no level3 events when error happens: ' + str(num_no_level3))
            with open(args.output_path + args.year + '/errors_level3/' + str(idx), 'a') as ferror:
                ferror.write('\n---\n')
                ferror.write(str(e))

    # json.dump(all_msgs, open('./outputs/test_msg_1-15_level3.json', 'w'), indent=4)
    # json.dump(all_rsps, open('./outputs/test_rsp_1-15_level3.json', 'w'), indent=4)

    print('num of no level3 events: ' + str(num_no_level3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--message", type=str, default="Hello! Who are you?")

    parser.add_argument('--year', type=str, default='2022',
                        help='specify the year of data to extract')
    parser.add_argument('--input_path', type=str, default='./data_cleaned/',
                        help='input data directory')
    parser.add_argument('--output_path', type=str, default='/ssd1/ccye/vicuna-event/',
                        help='output data directory')
    parser.add_argument('--alias', type=str, default='',
                        help='parsing results data directory, e.g. _remain')
    parser.add_argument('--start_rowid', type=int, default=0,
                        help='specify the start rowid in level2 df')
    parser.add_argument('--end_rowid', type=int, default=0,
                        help='specify the end rowid in level2 df')

    args = parser.parse_args()

    # specify the year of data
    year = args.year

    if not os.path.isdir(args.output_path + year + '/results_level3'):
        print('Making new dir: ' + args.output_path + year + '/results_level3')
        print('Making new dir: ' + args.output_path + year + '/errors_level3')
        os.makedirs(args.output_path + year + '/results_level3')
        os.makedirs(args.output_path + year + '/errors_level3')
    else:
        print('Dir exists: '+ args.output_path + year + '/results_level3')

    # load data
    print('start loading docs')
    data_path = args.input_path + 'doc_{}.json'.format(year)
    doc_list = json.load(open(data_path, 'r'))
    print('finish loading docs')

    # load level2 results
    level2_event_df = pd.read_csv('./results{}/level2/'.format(args.alias) + year + '_events.csv', sep='\t', dtype='string')
    print('\nNum of level2 parsed results in year ' + year + ':' + str(len(level2_event_df)))

    start_rowid = args.start_rowid
    end_rowid = args.end_rowid
    if end_rowid == 0 or end_rowid > len(level2_event_df) - 1:
        end_rowid = len(level2_event_df) - 1

    # check results
    exists_results = set(os.listdir(args.output_path + year + '/results_level3'))
    print('\nNum of exists level3 responses in year ' + year + ':' + str(len(exists_results)))


    dict_id2ont = json.load(open('./data/dict_id2ont_choice.json', 'r'))
    dict_hier_id = json.load(open('./data/dict_hier_id_level23.json', 'r'))

    generator = PromptGenerator(dict_id2ont, dict_hier_id)

    main(args, generator, doc_list, level2_event_df, exists_results, dict_hier_id, start_rowid, end_rowid)
