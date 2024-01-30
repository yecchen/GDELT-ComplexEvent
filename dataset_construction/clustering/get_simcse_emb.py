# from simcse import SimCSE
import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

device = 'cuda:0' if torch.cuda.is_available() else "cpu"
print('use device: ' + device)

def main(args, start_idx, end_idx, doc_list):

    # model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
    # model = SimCSE("princeton-nlp/sup-simcse-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large").to(device)

    # Tokenize input texts
    for idx, md5 in tqdm(enumerate(md5_list), total = len(md5_list)):
        if idx < start_idx or idx > end_idx:
            continue

        texts = [
            doc_list[md5]['Title'].strip(' \n\t'),
            '\n'.join(doc_list[md5]['Text'])
        ]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device) # 512
        # print(len(inputs))
        # print(len(inputs[0]))
        # print(len(inputs[1]))

        # Get the embeddings
        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            # print(embeddings.shape)
            doc_embedding = torch.mean(embeddings, dim=0)  # 1024,
            # print(doc_embedding.shape)
            torch.save(doc_embedding, args.output_path + md5 + '.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, default=0,
                        help='specify the start idx of md5')
    parser.add_argument('--end_idx', type=int, default=0,
                        help='specify the end idx of md5')
    parser.add_argument('--input_path', type=str, default='./all_vicuna_md5_list.json',
                        help='input data path')
    parser.add_argument('--output_path', type=str, default='/ssd1/ccye/doc_embs/',
                        help='output data directory')

    args = parser.parse_args()

    md5_list = json.load(open(args.input_path, 'r'))
    print('Number of docs in md5_list: ' + str(len(md5_list)))

    if not os.path.isdir(args.output_path):
        print('Making new dir: ' + args.output_path)
        os.makedirs(args.output_path)
    else:
        print('Dir exists: '+ args.output_path)

    print('start loading docs')
    data_path = './all_vicuna_docs.json'
    doc_list = json.load(open(data_path, 'r'))
    print('Number of docs in docs json: ' + str(len(doc_list)))
    print('end loading docs')

    start_idx = args.start_idx
    end_idx = args.end_idx
    if end_idx == 0 or end_idx > len(md5_list) - 1:
        end_idx = len(md5_list) - 1

    main(args, start_idx, end_idx, doc_list)







