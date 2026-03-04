from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoConfig, AutoModelForCausalLM, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import time


OPTIMIZERS = ("adamw", "muon", "mezo", "hybrid")

MODEL_PATH = './Qwen2.5-0.5B'

ds = load_dataset("Elriggs/openwebtext-100k", split='train[:6%]')

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
print(sequence_length)
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

def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

def muon_update(grad, momentum, beta = 0.95, ms_steps = 5):
    momentum.lerp_(grad, 1 - beta)
    update = momentum
    if update.ndim >2:
        update = update.view(update.size(0), -1)
    elif update.ndim < 2:
        update = update.view(1,-1)
    update = newtonschulz5(update, steps=ms_steps)
    update*=max(1,update.size(-2) / update.size(-1))**0.5
    return update

class MeZO(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, eps=1e-3):
        defaults = dict(lr=lr, eps=eps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, model, batch):

        group = self.param_groups[0]
        lr = group['lr']
        eps = group['eps']

        r_seed = torch.randint(0, 2**32, (1,)).item()
        def  add_noise(scale):
            torch.manual_seed(r_seed)
            for group in self.param_groups:
                for p in group['params']:
                    z = torch.randn_like(p)
                    p.add_(z, alpha= scale)
        add_noise(eps)
        L1 = model(**batch).loss
        add_noise(eps * -2)
        L2 = model(**batch).loss
        add_noise(eps)
        grad = (L1-L2)/(2*eps)
        torch.manual_seed(r_seed)
        for group in self.param_groups:
            for p in group['params']:
                z = torch.randn_like(p)
                p.add_(z, alpha=-lr * grad)
        return (L2 + L1) / 2

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr = 0.02, weight_decay = 0, momentum = 0.95):
        defaults = dict(lr = lr, weight_decay = weight_decay, momentum = momentum)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta = group["momentum"])
                p.mul_(1-group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha =-group["lr"])
        return loss



model = get_peft_model(model, _Loraconfig)

# with torch.no_grad():
#     for name, param in model.named_parameters():
#         if 'lora_B' in name:
#             param.normal_(std=1e-6)

model.to(device=device)
model.gradient_checkpointing_enable()
muon_params = []
adam_params = []

for n,p in model.named_parameters():
    if p.requires_grad:
        if p.ndim >=2:
            muon_params.append(p)
        else:
            adam_params.append(p)

optim_muon = Muon(muon_params, lr = 5e-4, momentum=0.95, weight_decay=0.01)
if adam_params:
    optim_adam = torch.optim.AdamW(adam_params,lr = 1e-5)

train_dataset = preprocessing()
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
micro_batch_size = 4
logical_batch_size = 32
accumulation_steps = logical_batch_size // micro_batch_size
train_loader = DataLoader(train_dataset,batch_size=micro_batch_size, shuffle=True, collate_fn=data_collator, num_workers=4,pin_memory=True,prefetch_factor=2)



def setup_optimizer(model, args, total_steps):
    if args.optimizer == "mezo":
        return [MeZO(model,lr=args.lr,eps=args.eps), None]

    emuon_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    adam_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    
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
    
    return [optimizers,scheduler]




optimizer_obj, scheduler = setup_optimizer(model=model, args = args, total_steps=total_steps)





model.train()
epochs = 1

history = {"loss": []}

total_steps = (len(train_loader) // accumulation_steps) * epochs
warmup_steps = int(0.1 * total_steps)

scheduler_muon = get_linear_schedule_with_warmup(
    optim_muon,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

for epoch in range(epochs):
    pbar = tqdm(train_loader, desc = f"Epoch:{epoch}")
    for i, batch in enumerate(pbar):
        batch = {k: v.to(device) for k,v in batch.items()}
        t0 = time.time()
        if isinstance(optimizer_obj, MeZO):
             torch.cuda.reset_peak_memory_stats()
             raw_loss_item = optimizer_obj.step(model, batch)
             peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        else:
            outputs = model(**batch)
            raw_loss = outputs.loss
            loss = outputs.loss / accumulation_steps
            loss.backward()


            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optim_muon.step()
                if adam_params:
                    optim_adam.step()

                scheduler_muon.step()

                optim_muon.zero_grad()
                if adam_params:
                    optim_adam.zero_grad()
        pbar.set_postfix(loss = raw_loss.item())
        history["loss"].append(raw_loss.item())
        if len(history["loss"]) % 100 == 0:
            with open("train_history.json", "w") as f:
                json.dump(history, f)
        time.sleep(1)