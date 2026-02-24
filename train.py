import torch
import argparse
import os
from models.bitnet_llama import BitLLaMA
from models.bitnet_transformer import BitNetLM
from data.data import get_data_and_vocab

def main():
    parser = argparse.ArgumentParser(description="BitNet Training Script")
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'transformer'])
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='모델을 저장할 폴더명')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_iters', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()

    # 1. 저장 폴더 생성 로직
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created directory: {args.save_dir}")

    # 2. 데이터 준비 및 장치 설정
    data, vocab_size, _, _ = get_data_and_vocab('vocab.pkl')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 3. 모델 초기화
    if args.model == 'llama':
        model = BitLLaMA(vocab_size).to(device)
    else:
        model = BitNetLM(vocab_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Training {args.model} on {device}...")
    for iter in range(args.max_iters):
        ix = torch.randint(len(data) - 128, (args.batch_size,))
        x = torch.stack([data[i:i+128] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+128+1] for i in ix]).to(device)

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % 100 == 0:
            print(f"[{args.model}] Iter {iter}: Loss {loss.item():.4f}")
    save_name = f"bitnet_{args.model}_lr{args.lr}_bs{args.batch_size}.pth"
    save_path = os.path.join(args.save_dir, save_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model successfully saved to: {save_path}")

if __name__ == "__main__":
    main()