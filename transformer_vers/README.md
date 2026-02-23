# BitNnet transformer

`BitLinear`를 적용하여 구현한 Character-level transformer 언어 모델입니다. [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) 데이터셋을 학습하여 셰익스피어 스타일의 텍스트를 생성합니다.

## 프로젝트 구조
- `utils.py`: 1.58-bit 행렬 곱셈을 모사하는 `BitLinear` 구현체
- `model.py`: BitSelfAttention 및 Transformer 블록 등 모델 아키텍처
- `train.py`: 모델 학습 및 가중치(`bitnet_model_tf_vers.pth`) 저장
- `generate.py`: 학습된 모델을 로드하여 텍스트 생성 추론

## Usage

### 모델 학습 (Training)
데이터를 자동으로 다운로드하고 10,000 이터레이션 동안 학습 진행합니다. 학습이 완료되면 `bitnet_model_tf_vers.pth`와 `vocab.pkl` 파일이 생성됩니다.
```
python train.py
```