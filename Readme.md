# Llama-3.2-1B-Instruct-korQuAD-v1

```
language:
- ko
license: llama3.2
base_model:
- meta-llama/Llama-3.2-1B-Instruct
datasets:
- KorQuAD/squad_kor_v1
```

이 프로젝트는 Llama-3.2-1B-Instruct를 기반으로 한국어 질의응답 태스크를 파인튜닝, 인퍼런스, 이밸류에이션할수 있는 프로젝트입니다.

## 모델 설명
- 기본 모델: Llama-3.2-1B-Instruct
- 학습 데이터셋: KorQuAD v1.0
- 학습 방법: LoRA (Low-Rank Adaptation)
- 주요 태스크: 한국어 질의응답

## 버전 히스토리
### v1.0.0(2024-10-02)
- 초기 버전 업로드
- KorQuAD v1.0 데이터셋 파인튜닝

### v1.1.0(2024-10-30)
- 모델 프롬프트 및 학습 방법 개선
- KorQuAD evaluate 코드 적용

## 성능
| 모델 | Exact Match | F1 Score |
|------|-------------|----------|
| Llama-3.2-1B-Instruct-v1 | 18.86 | 37.2 |
| Llama-3.2-1B-Instruct-v2 | 36.07 | 59.03 |
※ https://korquad.github.io/category/1.0_KOR.html의 evaluation script 사용

## 코드 설명
```
1. fine_tuning.py
- 모델 파인튜닝 코드
- training_params 변경하여 학습 하이퍼파라미터 수정 가능

2. inference.py
- 모델 인퍼런스 코드
- 모델 경로 변경 후 사용 가능

3. evaluation.py
- 모델 이밸류에이션 코드
- KorQuAD evaluate를 위한 json 저장

4. evaluate-v1.0.py
- KorQuAD evaluate 코드 적용
- 평가 지표 출력(Exact Match, F1 Score)
$python evaluate-v1.0.py [dataset_file] [prediction_file]
```

## 학습 세부 정보
- step: 2000
- 배치 크기: 1
- 학습률: 2e-4
- 옵티마이저: AdamW (32-bit)
- LoRA 설정:
  - r: 16
  - lora_alpha: 16
  - 대상 모듈: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
  - lora_dropout: 0.01

## 연락처
- njsung1217@gmail.com
- https://github.com/nakjun
- https://huggingface.co/NakJun/Llama-3.2-1B-Instruct-korQuAD-v1