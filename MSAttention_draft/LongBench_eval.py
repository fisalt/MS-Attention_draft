from datasets import load_dataset
import os
os.environ['CURL_CA_BUNDLE'] = ''
datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", \
            "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

data_dir="/home/wangning/LongBench"
val_files = [os.path.join(root, file) for root, _, files in os.walk(os.path.join(data_dir, 'data')) for file in files if file.endswith('.jsonl')]
data = load_dataset('json', data_files=val_files)
print(data)
for dataset in datasets:
    # data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test',cache_dir="/home/wangning/LongBench")
    data = load_dataset('json', data_files="/home/wangning/LongBench/data/"+f"{dataset}_e.jsonl")['train']
    # data = load_dataset(data_dir='/home/wangning/LongBench/data/',data_files=f"{dataset}_e", split='test')
    print(data)