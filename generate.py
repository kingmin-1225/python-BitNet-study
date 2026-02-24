import os
import argparse
import torch
import pickle
from models.bitnet_llama import BitLLaMA
from models.bitnet_transformer import BitNetLM

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('vocab.pkl', 'rb') as f:
        vocab_info = pickle.load(f)
    
    stoi = vocab_info['stoi']
    itos = vocab_info['itos']
    vocab_size = vocab_info['vocab_size']

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    parser = argparse.ArgumentParser(description="BitNet Training Script")
    parser.add_argument('--model_type', type=str, default='llama', choices=['llama', 'transformer'])
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='모델을 저장할 폴더명')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_iters', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()
    
    save_dir = 'checkpoints'
    save_name = f"bitnet_{args.model_type}_lr{args.lr}_bs{args.batch_size}.pth"
    model_path = os.path.join(save_dir, save_name)

    # 모델 초기화
    if args.model_type == 'llama':
        model = BitLLaMA(vocab_size).to(device)
    else:
        model = BitNetLM(vocab_size).to(device)

    # 가중치 로드
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded model from: {model_path}")
    else:
        print(f"Error: Model file not found at {model_path}")
        return

    model.eval() # 추론 모드 전환

    start_text = "First Citizen:"
    context = torch.tensor(encode(start_text), dtype=torch.long, device=device).unsqueeze(0)

    print("--- Generating Text ---")
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=256)
    
    print(decode(generated[0].tolist()))

if __name__ == "__main__":
    main()