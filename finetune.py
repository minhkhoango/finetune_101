# pip install transformers datasets torch accelerate
# pip install transformers datasets peft accelerate bitsandbytes 
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

################################################## INITIALIZATION ################################################


# Intialize the accelerator
accelerator = Accelerator(
    gradient_accumulation_steps=8,
    mixed_precision="fp16",
    log_with="tensorboard",
    project_dir="./logs"
)
device = accelerator.device

print(f'Using {torch.cuda.device_count()} GPUs via Accelerate')

# Configure quantization properly
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
)

model_name = "VietnamAIHub/Vietnamese_llama_30B_SFT" #"vilm/vinallama-12.5b-chat-DUS"  #"vilm/vinallama-7b-chat" 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0}, #device_map="auto",
    torch_dtype=torch.float16,
)


tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set padding_token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Get PEFT model
model = get_peft_model(model, peft_config)


######################################################## PREPROCESS DATA #######################################################


dataset = load_dataset("json", data_files="trimmed_data.json", split='train')
train_dataset = dataset.train_test_split(test_size=0.001)["train"]
eval_dataset = dataset.train_test_split(test_size=0.2)["test"]

MAX_LENGTH = 2048
# Improve preprocessing for Vietnamese legal text
def preprocess(batch):
    inputs = []
    for inp, out in zip(batch["input"], batch["output"]):
        prompt, completion = f"User: {inp}\nAssistant: ", out
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + completion_ids + [tokenizer.eos_token_id]
        input_ids = input_ids[:MAX_LENGTH] 

        # Truncate if the length exceeds max length
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]

        labels = [-100] * len(prompt_ids) + completion_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        inputs.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
    return {k: [dic[k] for dic in inputs] for k in inputs[0]}

tokenized_train = train_dataset.map(
    preprocess,
    batched=True,
    remove_columns=train_dataset.column_names
)

tokenized_eval = eval_dataset.map(
    preprocess,
    batched=True,
    remove_columns=eval_dataset.column_names
)

train_loader = DataLoader(tokenized_train, shuffle=True, batch_size=8)
eval_loader = DataLoader(tokenized_eval, shuffle=False, batch_size=8)

# Function to pad batches to maximum length
def collator(features):
    max_len = max(len(f["input_ids"]) for f in features)
    pad = lambda x, pad_val: [x + [pad_val] * (max_len - len(x)) for x in x]
    return {k: torch.tensor(pad([f[k] for f in features], tokenizer.pad_token_id if k != "labels" else -100)) 
            for k in ["input_ids", "attention_mask", "labels"]}

############################################################### END ##########################################################


############################################################# TRAINING #######################################################

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Prepare loaders
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=collator)
eval_loader = DataLoader(eval_dataset, batch_size=4, collate_fn=collator)

# Prepare for multi-GPU
model, optimizer, train_loader, eval_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader)

# Training loop
model.train()
for epoch in range(10):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        outputs = model(**batch)
        loss = outputs.loss / accelerator.gradient_accumulation_steps
        accelerator.backward(loss)

        if accelerator.sync_gradients:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} loss: {total_loss:.4f}")

# Save
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("lora_finetuned_model")
tokenizer.save_pretrained("lora_finetuned_model")
################################################## END ###################################

########################################## INFERENCE #####################################

# Inference
def generate_response(input_text):
    if accelerator.is_main_process:
        model.eval()

    prompt = f"User: {input_text}"
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs.input_ids, max_length=512, 
        num_beams=4, do_sample=True, temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Interactive loop
if __name__ == "__main__":
    while True:
        query = input("Enter your query (type ' exit' to quit)")
        if query.lower() == 'exit': break
        print(f'\nResponse: {generate_response(query)}')