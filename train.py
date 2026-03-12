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

MODEL_PATH = './Qwen3-4B-Thinking-2507'

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

sequence_length = 2048
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
model.print_trainable_parameters()


def setup_optimizer(model, args, total_steps):
    if args.optimizer == "mezo":
        return [MeZO(model.parameters(),lr=args.lr,eps=args.eps)], [None]

    muon_params = []
    adam_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim>=2 and ("lora_A" in name):
            muon_params.append(p)
        else:
            adam_params.append(p)
    
    print("Len of muon params: ", len(muon_params))

    optimizers = []
    schedulers = []
    
    if args.optimizer == "muon":
        optimizers.append(Muon(model.parameters(), lr=args.lr))
        scheduler = get_linear_schedule_with_warmup(optimizers[0], int(0.04*total_steps), total_steps)
        schedulers.append(scheduler)
    
    elif args.optimizer == "adamw":
        optimizers.append(torch.optim.AdamW(model.parameters(), lr=args.lr))
        scheduler = get_linear_schedule_with_warmup(optimizers[0], int(0.04*total_steps), total_steps)
        schedulers.append(scheduler)

    elif args.optimizer == "hybrid":
        if muon_params:
            optimizers.append(Muon(muon_params, lr=args.lr))
        if adam_params:
            optimizers.append(torch.optim.AdamW(adam_params, lr=args.lr * 0.1))
        
        scheduler = get_linear_schedule_with_warmup(optimizers[0], int(0.04*total_steps), total_steps)
        scheduler2 = get_linear_schedule_with_warmup(optimizers[1], int(0.04*total_steps), total_steps)
        schedulers.append(scheduler)
        schedulers.append(scheduler2)
    
    return optimizers,schedulers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="adamw", choices=OPTIMIZERS)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()


    history = {
    "config": vars(args),
    "lora_config": _Loraconfig.to_dict(),
    "loss": [],
    "stats": [],
    "eval_results": {}
}
    
    model.to(device=device)
    if args.optimizer != "mezo":
        model.gradient_checkpointing_enable()

    train_dataset = preprocessing()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    micro_batch_size = 1
    logical_batch_size = 32
    accumulation_steps = logical_batch_size // micro_batch_size
    train_loader = DataLoader(train_dataset,batch_size=micro_batch_size, shuffle=True, collate_fn=data_collator, num_workers=2,pin_memory=True,prefetch_factor=2)

    total_steps = (len(train_loader) // accumulation_steps) * args.epochs
    warmup_steps = int(0.03 * total_steps)

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
                    for sched in scheduler:
                        sched.step()

                peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            
            step_time = time.time() - t0

            tokens_per_sec = (micro_batch_size * sequence_length) / step_time

            history["loss"].append(current_loss)

            if args.optimizer == "hybrid":
                lr_muon = scheduler[0].get_last_lr()[0]
                lr_adam = scheduler[1].get_last_lr()[0]
                current_lr = f"M:{lr_muon:.2e}|A:{lr_adam:.2e}"
            else:
                current_lr = f"{scheduler[0].get_last_lr()[0]:.2e}" if (scheduler and scheduler[0]) else f"{args.lr:.2e}"

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
                with open(f"history_{args.optimizer}.json", "w") as f:
                    # default=str заставит JSON превращать сеты и другие объекты в строки
                    json.dump(history, f, indent=4, default=str)
            time.sleep(0.5)

    model.save_pretrained(f"qwen_lora_{args.optimizer}")

    print("Piqa evaluating")

    import subprocess

    eval_cmd = [
        "python3", "-m", "lm_eval", "run",  # Добавили 'run'
        "--model", "hf",
        "--model_args", f"pretrained={MODEL_PATH},peft=./qwen_lora_{args.optimizer}",
        "--tasks", "piqa",
        "--device", "cuda:0",
        "--batch_size", "auto"
    ]
    try:
        result = subprocess.run(eval_cmd, capture_output=True, text=True)
        # Парсим вывод (упрощенно) или просто сохраняем текст
        with open(f"eval_{args.optimizer}.txt", "w") as f:
            f.write(result.stdout)
        print(f"Evaluating completed. Results are in eval_{args.optimizer}.txt")
    except Exception as e:
        print(f"PIQA error: {e}")


if __name__ == "__main__":
    main()