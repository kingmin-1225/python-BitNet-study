import torch
from model import BitNetLM
from data import get_data_and_vocab

# 1. 데이터 준비
data, vocab_size, _, _ = get_data_and_vocab('vocab.pkl')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. 모델 초기화
model = BitNetLM(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# 3. 학습 루프
print("Training started...")
for iter in range(10000):
    ix = torch.randint(len(data) - 256, (32,))
    x = torch.stack([data[i:i+256] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+256+1] for i in ix]).to(device)

    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if iter % 100 == 0:
        print(f"Iter {iter}: Loss {loss.item():.4f}")

# 4. 모델 가중치 저장
torch.save(model.state_dict(), 'bitnet_model_tf_vers.pth')
print("Model saved to bitnet_model.pth")