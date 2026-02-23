import torch
import pickle
from model import BitNetLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 단어장 불러오기
with open('vocab.pkl', 'rb') as f:
    vocab_info = pickle.load(f)
stoi = vocab_info['stoi']
itos = vocab_info['itos']
vocab_size = vocab_info['vocab_size']

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 2. 모델 불러오기
model = BitNetLM(vocab_size).to(device)
model.load_state_dict(torch.save('bitnet_model.pth', weights_only=True))
model.eval() # 추론 모드 전환

# 3. 텍스트 생성
start_text = "First Citizen:"
context = torch.tensor(encode(start_text), dtype=torch.long, device=device).unsqueeze(0)
generated = model.generate(context, max_new_tokens=512)
print("--- Generated Text ---")
print(decode(generated[0].tolist()))