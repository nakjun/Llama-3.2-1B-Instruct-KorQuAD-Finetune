from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer


def formatting_prompts_func(examples):
    eos_token = "respond"
    korQuAD_prompt = """<s>[INST] 다음 문맥을 바탕으로 질문에 답해주세요.

    문맥:
    {1}

    질문: {0} [/INST]

    {2}</s>
    """

    instructions = examples["question"]
    inputs = examples["context"]
    outputs = [item["text"][0] for item in examples["answers"]]
    texts = []

    for instruction, input, output in zip(instructions, inputs, outputs):
        text = korQuAD_prompt.format(instruction, input, output) + eos_token
        texts.append(text)

    return {"text": texts}


# model_path = 'meta-llama/Llama-3.2-1B-Instruct'   #huggingface에 연결하는 case
model_path = "./models/Llama-3.2-1B-Instruct"  # 로컬에서 model을 불러오는 case

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = "respond"

dataset = load_dataset("KorQuAD/squad_kor_v1", split="train")
dataset = dataset.shuffle(seed=42).select(range(7500))  # 데이터셋을 섞고 3만 개로 제한
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)
dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])
print(dataset[:1])

# # 모델 준비
model = prepare_model_for_kbit_training(model)

# LoRA 설정
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ],
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
)

# # LoRA 적용
model = get_peft_model(model, lora_config)

training_params = TrainingArguments(
    output_dir="/results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=2000,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()
ADAPTER_MODEL = "lora_adapter_test1"
trainer.model.save_pretrained(ADAPTER_MODEL)
