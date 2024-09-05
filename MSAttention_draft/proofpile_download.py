from datasets import load_dataset
import torch
# dataset = load_dataset("hoskinson-center/proof-pile",cache_dir="./proofpile",split="test")
# dataset = load_dataset('pg19',cache_dir="/home/wangning/pg19/datasets")
datapath="/home/wangning/pg19/datasets"
dataset = load_dataset('pg19',cache_dir=datapath)
print(dataset['train'])