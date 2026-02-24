# Python-BitNet-Study: 1.58-bit LLM Optimization
> **BitNet 논문을 바탕으로 한 LLM 경량화 및 아키텍처 비교 구현 프로젝트**

본 프로젝트는 Microsoft의 "The Era of 1-bit LLMs" 논문을 바탕으로, **BitLinear** 기법을 기존의 Transformer와 최신 Llama 구조에 각각 적용하여 구현한 토이 프로젝트입니다.

## Key Features
- **BitLinear Implementation**: 
    - **Ternary Weight Quantization**: 가중치를 ${-1, 0, 1}$로 양자화하여 연산 효율성을 극대화했습니다.
    - **8-bit Activation Quantization**: 추론 시 활성화 함수를 8-bit로 양자화하여 메모리 대역폭을 절약합니다.
    - **STE 기법 적용**: 양자화 함수는 미분이 불가능하므로, `(quant - original).detach() `패턴을 사용하여 순전파 시에는 양자화된 값을 사용하고, 역전파 시에는 고정밀도 가중치에 그래디언트가 전달되도록 설계했습니다.
- **Architecture Comparison**: 
  - `Vanilla Transformer`: 표준 트랜스포머 구조에서의 BitNet 적용.
  - `Llama-based BitNet`: RoPE(Rotary Positional Embedding), RMSNorm, SwiGLU 등 최신 LLM 기법과 BitNet의 결합.
- **Modular Design**: 모델, 데이터 로더, 학습 로직을 분리하여 확장성 있는 코드 구조 설계.
- **CLI Support**: `argparse`를 통한 유연한 실험 환경 제어.

## Tech Stack
- **Language**: Python 3.11+
- **Framework**: PyTorch
- **Hardware**: NVIDIA RTX 4060 Ti (8GB VRAM)
- **Dataset**: Shakespeare (Tiny-scale), WikiText (Planned)

## Project Structure
```text
.
├── models/
│   ├── bitnet_llama.py        # Llama-based 1.58-bit implementation
│   └── bitnet_transformer.py  # Vanilla Transformer-based 1.58-bit implementation
├── data/
│   ├── data.py                # Data preprocessing & Vocab
│   └── vocab.pkl              # Tokenizer vocabulary file
├── checkpoints/               # Directory for saved model weights
├── utils.py                   # BitLinear implementation
├── train.py                   # Unified training script
├── generate.py                # Text generation/inference script
└── README.md
```

## Usage
- **Train**
  ```
  python train.py
  ```
  - `--model`: 학습할 모델 선택 (기본값: `llama`)
  - `--save_dir`: 체크포인트 저장 경로 (기본값: `checkpoints`)
  - `--batch_size`: batch_size (기본값: `32`)
  - `--max_iters`: 반복 횟수 (기본값: `3000`)
  - `--lr`: 학습률 (기본값: `2e-4`)
- **Generate**
  ```
  python generate.py
  ```
  - `--model`: 사용할 모델 선택 (기본값: `llama`)
  - `--save_dir`: 체크포인트 저장 경로 (기본값: `checkpoints`)
  - `--batch_size`: batch_size (기본값: `32`)
  - `--lr`: 학습률 (기본값: `2e-4`)

## Comparison
| Model | Param Count | VRAM Usage | Training Loss (3k iters) |
| :--- | :---: | :---: | :---: |

### Papers
- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)

