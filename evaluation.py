from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from evaluate import load
import pandas as pd
import os
import json
from peft import PeftModel

# 데이터셋 로드
validation_dataset = load_dataset("KorQuAD/squad_kor_v1", split="validation")
validation_dataset = validation_dataset.shuffle(seed=42)
print(validation_dataset)

# 모델 및 토크나이저 로드
model_path = "models/Llama-3.2-1B-Instruct"
ADAPTER_MODEL = "lora_adapter_test1"
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="cuda:2"
)
model = PeftModel.from_pretrained(
    model, ADAPTER_MODEL, device_map="auto", torch_dtype=torch.bfloat16
)
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(model_path)


# 평가 함수
def evaluate_model(model, tokenizer, dataset):
    predictions = []
    references = []

    for i, example in enumerate(tqdm(dataset)):
        context = example["context"]
        question = example["question"]
        answers = example["answers"]

        korQuAD_prompt = f"""<s>[INST] 다음 문맥을 바탕으로 질문에 답해주세요.

문맥:
{context}

질문: {question} [/INST]
"""

        input_ids = tokenizer.encode(korQuAD_prompt, return_tensors="pt").to(
            model.device
        )

        output = model.generate(
            input_ids,
            max_new_tokens=256,
            temperature=0.05,
            repetition_penalty=1.3,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

        result = tokenizer.decode(output[0], skip_special_tokens=True)
        prediction = result.split("[/INST]")[-1].split("</s>")[0].strip()

        predictions.append({"id": str(example["id"]), "prediction_text": prediction})
        references.append({"id": str(example["id"]), "answers": answers})

        print(question)
        print(prediction, " vs ", answers)

    return predictions, references


# 평가 실행
predictions, references = evaluate_model(model, tokenizer, validation_dataset)

# 평가 지표 계산
squad_metric = load("squad")
results = squad_metric.compute(predictions=predictions, references=references)

print(f"Exact Match: {results['exact_match']}")
print(f"F1 Score: {results['f1']}")

# 결과를 DataFrame으로 변환
df = pd.DataFrame(
    {
        "ID": [p["id"] for p in predictions],
        "질문": [example["question"] for example in validation_dataset],
        "컨텍스트": [example["context"] for example in validation_dataset],
        "예측 답변": [p["prediction_text"] for p in predictions],
        "실제 답변": [
            r["answers"]["text"][0] if r["answers"]["text"] else "" for r in references
        ],
    }
)

# 최종 결과 저장 (Excel)
excel_filename = "evaluation_results_final.xlsx"
df.to_excel(excel_filename, index=False, engine="openpyxl")
print(f"최종 결과가 {excel_filename} 파일로 저장되었습니다.")

# 최종 예측 결과를 JSON으로 저장
final_json_filename = "predictions_final.json"
formatted_predictions = {pred["id"]: pred["prediction_text"] for pred in predictions}
with open(final_json_filename, "w", encoding="utf-8") as f:
    json.dump(formatted_predictions, f, ensure_ascii=False, indent=2)
print(f"최종 예측 결과가 {final_json_filename} 파일로 저장되었습니다.")
