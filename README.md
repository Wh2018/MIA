## Paper Overview

**Remembering Facts, Updating Minds: Personal-Factual Adaptive Memory for
Multi-Turn Emotional Support**

This project studies multi-turn Emotional Support Conversation systems. Existing
methods often pass inferred seeker information to the generator as a single
undifferentiated state, which makes persistent facts and transient mental states
hard to store, select, and update differently. This can weaken factual grounding
and cause the model to rely on stale psychological assumptions across turns.

MIA is a Personal-Factual Adaptive Memory framework. It first infers Emotional
Theory-of-Mind factors from the dialogue, separates them into factual and
personal states, uses a Personal-Factual Discriminator to decide which state
should guide the next response, and applies an Obsolescence Removal Mechanism to
remove outdated memory before generation. The goal is to preserve stable factual
grounding while adapting to changes in the seeker's psychological state.

The experiments are designed for ESConv and CPsyCounD. The code supports EToM
annotation, PFD labeling, adaptive memory construction, instruction-data
conversion, LoRA fine-tuning, baseline prediction, and LLM-as-a-judge
evaluation.


## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install LLaMA-Factory separately if you want to run LoRA training and
prediction:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
pip install -e '.[torch,metrics]'
```

## Required Files

Check `config/llm_api.yaml`:

- `base_url`, `api_key`, `model_gen`: OpenAI-compatible endpoint for EToM and
  PFD generation.
- `model_judge`: model used by the memory-update ORM step.
- `eval_base_url`, `eval_api_key`, `model_eval`: optional endpoint for
  LLM-as-a-judge evaluation.
- `predict_base_url`, `predict_api_key`, `model_predict`: optional endpoint for
  API baseline prediction.
- `sbert_model_path`, `sbert_device`: local or Hugging Face embedding model used
  for memory similarity and exemplar retrieval.

API endpoints and keys are blank by default. Model names are public defaults and
can be changed to match your serving environment.

Check `config/training.yaml`:

- `base_model_path`: local base model for LoRA fine-tuning.
- `llamafactory_dir`: local LLaMA-Factory checkout.
- `cuda_device`: GPU id used by the scripts.

The base model is set to a public Hugging Face model id by default. Set
`llamafactory_dir` to your local LLaMA-Factory checkout before training.

Dataset locations:

- ESConv input: `data/raw/esc/esc.json` 
- CPsyCounD input: `data/raw/cpsy/cpsy.json` 

Original dataset sources:

- ESConv comes from **Towards Emotional Support Dialog Systems**. Paper:
  <https://aclanthology.org/2021.acl-long.269/>. Official data/code
  repository: <https://github.com/thu-coai/Emotional-Support-Conversation>.
  The released corpus file is `ESConv.json`.
- CPsyCounD comes from **CPsyCoun: A Report-based Multi-turn Dialogue
  Reconstruction and Evaluation Framework for Chinese Psychological
  Counseling**. Paper: <https://aclanthology.org/2024.findings-acl.830/>.
  Official repository:
  <https://github.com/CAS-SIAT-XinHai/CPsyCoun>. The LLaMA-Factory-format
  release is available at
  <https://huggingface.co/datasets/CAS-SIAT-XinHai/CPsyCoun>.

The exemplar banks `config/etom_exemplars_esc.json` and
`config/etom_exemplars_cpsy.json` are included. Each entry contains `seeker`,
`supporter`, `Belief`, `Intention`, `Desire`, `Emotion`, `Fact`, `Cause`, and
`Result`.

## Run

Build EToM, PFD, memory, and instruction data:

```bash
bash scripts/run_pipeline.sh
```

Run one dataset only:

```bash
RUN_CPSY=0 bash scripts/run_pipeline.sh
RUN_ESC=0 bash scripts/run_pipeline.sh
```

Run downstream conversion after EToM and PFD files already exist:

```bash
bash scripts/prepare_data.sh ESC
bash scripts/prepare_data.sh CPsy
```

Train and predict with LLaMA-Factory:

```bash
bash scripts/train.sh ESC
bash scripts/train.sh CPsy
```

Generate hosted-API baseline predictions:

```bash
python scripts/predict_api.py --dataset esc
python scripts/predict_api.py --dataset cpsy
```

Evaluate predictions with an LLM judge:

```bash
python scripts/evaluate_esc_llm_judge.py --help
python scripts/evaluate_cpsy_llm_judge.py --help
```

## Outputs

All generated artifacts are written under `data/`:

- `data/etom/`: raw and parsed seven-factor EToM annotations.
- `data/pfd/`: raw and parsed Personal-Factual Discriminator annotations.
- `data/memory/`: updated memory states and logs.
- `data/instruction/`: train/test instruction JSON files.
- `data/training/`: LoRA adapters, predictions, and evaluation outputs.
- `data/stats/`: optional statistics files.

