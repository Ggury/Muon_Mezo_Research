from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoConfig, AutoModelForCausalLM, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import time
import argparse

from Muon import Muon
from MeZO import MeZO


OPTIMIZERS = ("adamw", "muon", "mezo", "hybrid")

MODEL_PATH = './Qwen2.5-0.5B'

ds = load_dataset("Elriggs/openwebtext-100k", split='train[:6%]')

epochs = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
config = AutoConfig.from_pretrained(MODEL_PATH)

_Loraconfig = LoraConfig(
    r=32,
    lora_alpha = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
    )

sequence_length = 1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocessing():
    def tokenize(examples):
        return tokenizer(examples["text"], truncation = False)
    tk_dataset = ds.map(tokenize,batched=True,remove_columns=["text"])


    
    def group(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // sequence_length) * sequence_length

        result = {
            k: [t[i : i + sequence_length] for i in range(0, total_length, sequence_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    lm_dataset = tk_dataset.map(group, batched=True)
    return lm_dataset


model = get_peft_model(model, _Loraconfig)


def setup_optimizer(model, args, total_steps):
    if args.optimizer == "mezo":
        return [MeZO(model.parameters(),lr=args.lr,eps=args.eps)], None

    muon_params = []
    adam_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim>=2 and ("lora" in name):
            muon_params.append(p)
        else:
            adam_params.append(p)
    
    optimizers = []
    
    if args.optimizer == "muon":
        optimizers.append(Muon(model.parameters(), lr=args.lr))
        scheduler = get_linear_schedule_with_warmup(optimizers[0], int(0.1*total_steps), total_steps)
    
    elif args.optimizer == "adamw":
        optimizers.append(torch.optim.AdamW(model.parameters(), lr=args.lr))
        scheduler = get_linear_schedule_with_warmup(optimizers[0], int(0.1*total_steps), total_steps)

    elif args.optimizer == "hybrid":
        if muon_params:
            optimizers.append(Muon(muon_params, lr=args.lr))
        if adam_params:
            optimizers.append(torch.optim.AdamW(adam_params, lr=args.lr * 0.1))
        
        scheduler = get_linear_schedule_with_warmup(optimizers[0], int(0.1*total_steps), total_steps)
    
    return optimizers,scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="adamw", choices=OPTIMIZERS)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eps", type=float, default=1e-3) # Для MeZO
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()


    history = {
    "config": vars(args), # Сохраняем все гиперпараметры из argparse
    "lora_config": _Loraconfig.to_dict(),
    "loss": [],
    "stats": [],
    "eval_results": {} # Сюда запишем результат PIQA позже
}
    
    model.to(device=device)
    if args.optimizer != "mezo":
        model.gradient_checkpointing_enable()

    train_dataset = preprocessing()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    micro_batch_size = 4
    logical_batch_size = 32
    accumulation_steps = logical_batch_size // micro_batch_size
    train_loader = DataLoader(train_dataset,batch_size=micro_batch_size, shuffle=True, collate_fn=data_collator, num_workers=4,pin_memory=True,prefetch_factor=2)

    total_steps = (len(train_loader) // accumulation_steps) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    optimizer_obj, scheduler = setup_optimizer(model=model, args = args, total_steps=total_steps)

    model.train()

    #history = {"loss": []}

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc = f"Epoch:{epoch}")
        for i, batch in enumerate(pbar):
            batch = {k: v.to(device) for k,v in batch.items()}
            t0 = time.time()

            torch.cuda.reset_peak_memory_stats()
            if isinstance(optimizer_obj[0], MeZO):
                current_loss = optimizer_obj[0].step(model, batch)
                peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            else:
                outputs = model(**batch)
                raw_loss = outputs.loss
                loss = outputs.loss / accumulation_steps
                loss.backward()
                current_loss = raw_loss.item()


                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                    for opt in optimizer_obj:
                        opt.step()
                        opt.zero_grad()

                    scheduler.step()

                peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            
            step_time = time.time() - t0

            tokens_per_sec = (micro_batch_size * sequence_length) / step_time

            history["loss"].append(current_loss)

            current_lr = scheduler.get_last_lr()[0] if scheduler else args.lr

            history["stats"].append({
                "time": step_time,
                "memory": peak_mem,
                "tokens_per_sec": tokens_per_sec,
                "lr": current_lr,
                "step": i
            })
        
            pbar.set_postfix(loss=f"{current_loss:.4f}", mem=f"{peak_mem:.0f}MB")
            # history["loss"].append(current_loss)
            # if "stats" not in history: history["stats"] = []
            # history["stats"].append({"time": step_time, "memory": peak_mem})

            if len(history["loss"]) % 100 == 0:
                with open(f"history{args.optimizer}.json", "w") as f:
                    json.dump(history, f)

    model.save_pretrained(f"qwen_lora_{args.optimizer}")

    print("Piqa evaluating")

    import subprocess

    eval_cmd = [
        "lm_eval", "--model", "hf",
        "--model_args", f"pretrained={MODEL_PATH},peft=qwen_lora_{args.optimizer}",
        "--tasks", "piqa",
        "--device", str(device),
        "--batch_size", "auto"
    ]

    try:
        result = subprocess.run(eval_cmd, capture_output=True, text=True)
        # Парсим вывод (упрощенно) или просто сохраняем текст
        with open(f"eval_{args.optimizer}.txt", "w") as f:
            f.write(result.stdout)
        print(f"✅ Оценка завершена. Результаты в eval_{args.optimizer}.txt")
    except Exception as e:
        print(f"❌ Ошибка при запуске PIQA: {e}")


if __name__ == "__main__":
    main()