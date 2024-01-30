"""
Usage:
python3 -m fastchat.serve.huggingface_api --model ~/model_weights/vicuna-7b/
"""
import argparse
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.conversation import get_default_conv_template
from fastchat.serve.inference import load_model, add_model_args

from tqdm import tqdm
import os

from fastchat.src_all.prompt_generator import PromptGenerator


@torch.inference_mode()
def main(args, generator, doc_list, md5_list, exists_results, start_idx, end_idx):
    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        debug=args.debug,
    )

    for idx, md5 in tqdm(enumerate(md5_list), total=len(md5_list)):
        if idx < start_idx or idx > end_idx:
            continue

        if md5 in exists_results:
            continue

        try:
            article = doc_list[md5]
            msg = generator.get_prompt_1_relation(article)

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


            with open(args.output_path + args.year + '/results/' + md5, 'w') as fresult:
                fresult.write(rsp)

        except Exception as e:
            with open(args.output_path + args.year + '/errors/' + md5, 'a') as ferror:
                ferror.write('\n---\n')
                ferror.write(str(e))


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
    parser.add_argument('--start_idx', type=int, default=0,
                        help='specify the start idx of md5')
    parser.add_argument('--end_idx', type=int, default=0,
                        help='specify the end idx of md5')
    parser.add_argument('--input_path', type=str, default='./data_cleaned/',
                        help='input data directory')
    parser.add_argument('--output_path', type=str, default='/ssd1/ccye/vicuna-event/',
                        help='output data directory')

    args = parser.parse_args()

    # specify the year of data
    year = args.year
    dict_year2md5 = json.load(open(args.input_path + 'dict_year2md5s.json'))
    md5_list = dict_year2md5[year]

    start_idx = args.start_idx
    end_idx = args.end_idx
    if end_idx == 0 or end_idx > len(md5_list) - 1:
        end_idx = len(md5_list) - 1

    if not os.path.isdir(args.output_path + year):
        print('Making new dir: ' + args.output_path + year)
        os.makedirs(args.output_path + year)
        os.makedirs(args.output_path + year + '/results')
        os.makedirs(args.output_path + year + '/errors')
    else:
        print('Dir exists: '+ args.output_path + year)

    # check results
    exists_results = set(os.listdir(args.output_path + year + '/results'))
    print('\nNum of exists results in year ' + year + ':' + str(len(exists_results)))

    # load data
    print('start loading docs')
    data_path = args.input_path + 'doc_{}.json'.format(year)
    doc_list = json.load(open(data_path, 'r'))
    print('finish loading docs')

    dict_id2ont = json.load(open('./data/dict_id2ont_choice.json', 'r'))
    dict_hier_id = json.load(open('./data/dict_hier_id.json', 'r'))

    generator = PromptGenerator(dict_id2ont, dict_hier_id)

    main(args, generator, doc_list, md5_list, exists_results, start_idx, end_idx)
