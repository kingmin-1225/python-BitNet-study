import requests
import torch
import pickle
import os

def get_data_and_vocab(save_path='data/vocab.pkl'):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    # 단어장 저장 (추론 시 사용)
    with open(save_path, 'wb') as f:
        pickle.dump({'stoi': stoi, 'itos': itos, 'vocab_size': vocab_size}, f)

    encode = lambda s: [stoi[c] for c in s]
    data = torch.tensor(encode(text), dtype=torch.long)
    
    return data, vocab_size, stoi, itos