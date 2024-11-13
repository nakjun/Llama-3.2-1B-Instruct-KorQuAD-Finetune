import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel, PeftConfig
import torch

# 모델 및 토크나이저 경로
BASE_MODEL = "./models/Llama-3.2-1B-Instruct"
ADAPTER_MODEL = "lora_adapter_test1"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# 기본 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, device_map="cuda:1", trust_remote_code=True
)

# LoRA 어댑터 로드 및 결합
peft_config = PeftConfig.from_pretrained(ADAPTER_MODEL)
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
model = model.merge_and_unload()

# 모델을 평가 모드로 설정
model.eval()

context = """	
비욘세는 텍사스 주 프레데릭스버그에 있는 세인트 메리 초등학교에 다니며 무용 수업을 들었다.
그녀의 노래 실력은 무용지도자인 다를렛 존슨이 노래를 흥얼거리기 시작하자 고음을 낼 수 있게 끝내면서 알게 됐다.비욘세의 음악과 공연에 대한 관심은 7살 때 학교 탤런트 쇼에서 우승한 뒤 존 레넌의 '상상'을 불러 15/16세 청소년들을 이겼다.
비욘세는 1990년 가을 휴스턴의 음악 자석학교인 파커초등학교에 입학해 이 학교의 합창단과 함께 공연할 예정이었다.그녀는 공연 및 시각예술 고등학교와 이후 알리에프 엘식 고등학교에 진학하기도 했다.비욘세는 2년간 솔리스트 자격으로 세인트 존스 유나이티드 감리교회 합창단원으로 활동하기도 했다.
"""

question = "비욘세가 합창단원으로 활동한 교회의 이름은?"

# 입력 형식 설정
korQuAD_prompt = f"""<s>[INST] 다음 문맥을 바탕으로 질문에 답해주세요.

문맥:
{context}

질문: {question} [/INST]
"""
# 토큰화
input_ids = tokenizer.encode(korQuAD_prompt, return_tensors="pt").to(model.device)

# 텍스트 생성
output = model.generate(
    input_ids,
    max_new_tokens=256,
    temperature=0.05,
    repetition_penalty=1.3,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
)

result = tokenizer.decode(output[0], skip_special_tokens=True)
answer = result.split("[/INST]")[-1].split("</s>")[0].strip()

print("생성된 답변:", answer)
