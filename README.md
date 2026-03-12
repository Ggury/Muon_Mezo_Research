# LLM Optimizer Benchmark
## Targets
Research the behavior of Muon, Hybrid of Muon and AdamW, Mezo optimizers on LoRa training of Qwen2.5-0.5B model.
## Experiment setup
**Model** : Qwen2.5-0.5B
**Dataset** : OpenWebText-100k (6% ~6000)

## Results

| Optimizer | Accuracy | VRAM usage |
|-----------|----------|------------|
| Baseline  | 0.695    | -          |
| **Hybrid**| **0.7013**| 5456 MB   |
| MeZO      | 0.7008   | **4030 MB**|
| AdamW     | 0.6991   | 5486 MB    |
| Muon      | 0.5800   | 6362 MB    |

## Setup
Clone repository and run
```bash
pip install -r requirements.txt
```
## Usage
In repository folder run 
```bash
python train.py --optimizer {adamw|muon|hybrid|mezo} --epochs 4 --lr 1e-4
```