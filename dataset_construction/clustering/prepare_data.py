import argparse
import json
import os
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from datetime import datetime

def main(args, docs, events):
    event_md5_list = list(events['Md5'].unique())
    json.dump(event_md5_list, open(args.output_path + 'md5_list.json', 'w'), indent=4)
    print('event md5 list length: ' + str(len(event_md5_list)))

    event_docs = {}
    for md5 in event_md5_list:
        event_docs[md5] = docs[md5]
    json.dump(event_docs, open(args.output_path + 'docs_title_paragraphs.json', 'w'), indent=4)
    print('event docs_title_paragraphs saved: ' + str(len(event_docs)))

    event_docs_list = []
    for idx, md5 in tqdm(enumerate(event_md5_list), total=len(event_md5_list)):
        doc = docs[md5]
        event_docs_list.append('\n'.join([doc['Title']] + doc['Text']))
    json.dump(event_docs_list, open(args.output_path + 'docs.json', 'w'), indent=4)
    print('event docs saved: ' + str(len(event_docs_list)))

    dates = list(events['Date'].unique())
    print('event dates: ' + str(len(dates)))
    mindate = datetime.strptime(dates[0], '%Y%m%d')
    date2nday = {}
    for date in dates:
        nday = (datetime.strptime(date, '%Y%m%d') - mindate).days
        date2nday[date] = nday
    json.dump(date2nday, open(args.output_path + 'date2nday.json', 'w'), indent=4)

    md52nday = {}
    events['ndays'] = [date2nday[x] for x in events['Date']]
    for idx, row in tqdm(events.iterrows(), total=len(events)):
        md52nday[row['Md5']] = row['ndays']
    json.dump(md52nday, open(args.output_path + 'md52nday.json', 'w'), indent=4)
    time_features = []
    maxnday = max(events['ndays'])
    print(maxnday)
    for md5 in event_md5_list:
        time_features.append(1.0 * md52nday[md5] / maxnday)
    time_embs = np.array(time_features)
    np.save(args.output_path + 'time_embs.npy', time_embs)
    print('time embeddings saved, shape: {}'.format(time_embs.shape))

    doc_emb_list = []
    for idx, md5 in tqdm(enumerate(event_md5_list), total=len(event_md5_list)):
        # all_emb_list.append(torch.load(doc_embed_path + '/' + md5 + '.pt', map_location=torch.device('cpu')))
        doc_emb_list.append(torch.load(args.doc_embed_path + md5 + '.pt', map_location=torch.device('cpu')))
    doc_embs = torch.stack(doc_emb_list).numpy()
    np.save(args.output_path + 'doc_embs.npy', doc_embs)
    print('doc embeddings saved, shape: {}'.format(doc_embs.shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/storage_fast/ccye/FastChat/data_final/EGIRIS_entlink.csv',
                        help='input csv data')
    parser.add_argument('--output_path', type=str, default='/storage_fast/ccye/FastChat/clustering/entlink/',
                        help='output path')
    parser.add_argument('--doc_embed_path', type=str, default='/ssd1/ccye/doc_embs/',
                        help='doc embed path')

    # args = parser.parse_args("")
    args = parser.parse_args()

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
        print('Make new dir: ' + args.output_path)
    else:
        print('Dir exists: '+ args.output_path)

    print('start loading all vicuna documents')
    docs = json.load(open('./all_vicuna_docs.json'))
    print('end loading all vicuna documents, number: ' + str(len(docs)))

    print('start loading event data')
    events = pd.read_csv(args.input_path, sep='\t', dtype='string')
    print('end loading event data, number: ' +  str(len(events)))

    main(args, docs, events)