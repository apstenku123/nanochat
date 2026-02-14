# nanochat

![nanochat logo](dev/nanochat.png)

> The best ChatGPT that $100 can buy.

This repo is a full-stack implementation of an LLM like ChatGPT in a single, clean, minimal, hackable, dependency-lite codebase. nanochat is designed to run on a single 8XH100 node via scripts like [speedrun.sh](speedrun.sh), that run the entire pipeline start to end. This includes tokenization, pretraining, finetuning, evaluation, inference, and web serving over a simple UI so that you can talk to your own LLM just like ChatGPT. nanochat will become the capstone project of the course LLM101n being developed by Eureka Labs.

## Talk to it

To get a sense of the endpoint of this repo, you can currently find [nanochat d34](https://github.com/karpathy/nanochat/discussions/314) hosted on [nanochat.karpathy.ai](https://nanochat.karpathy.ai/). "d34" means that this model has 34 layers in the Transformer neural network. This model has 2.2 billion parameters, it was trained on 88 billion tokens by simply running the training script [run1000.sh](run1000.sh) with `--target_param_data_ratio=40` (2x longer than Chinchilla-optimal), and the total cost of training was ~$2,500 (about 100 hours training time on 8XH100 GPU node). While today this is enough to outperform GPT-2 of 2019, it falls dramatically short of modern Large Language Models like GPT-5. When talking to these micro models, you'll see that they make a lot of mistakes, they are a little bit naive and silly and they hallucinate a ton, a bit like children. It's kind of amusing. But what makes nanochat unique is that it is fully yours - fully configurable, tweakable, hackable, and trained by you from start to end. To train and talk to your own, we turn to...

## Updates

- (Jan 7 2026) See new post: [nanochat Miniseries v1](https://github.com/karpathy/nanochat/discussions/420) and the associated script [miniseries.sh](miniseries.sh).

## Quick start

The fastest way to feel the magic is to run the speedrun script [speedrun.sh](speedrun.sh), which trains and inferences the $100 tier of nanochat. On an 8XH100 node at $24/hr, this gives a total run time of about 4 hours. Boot up a new 8XH100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

```bash
bash speedrun.sh
```

Alternatively, since the script runs for 4 hours, I like to launch it like this inside a new screen session `speedrun` (and also log output to `speedrun.log`):

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

See the [screen cheatsheet](https://gist.github.com/jctosta/af918e1618682638aa82) if you are less familiar. You can watch it go inside the screen session, or detach with `Ctrl-a d` and `tail speedrun.log` to view progress. Now wait 4 hours. Once it's done, you can talk to your LLM via the ChatGPT-like web UI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and serve it:

```bash
python -m scripts.chat_web
```

And then visit the URL shown. Make sure to access it correctly, e.g. on Lambda use the public IP of the node you're on, followed by the port, so for example [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/), etc. Then talk to your LLM as you'd normally talk to ChatGPT! Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :).

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

You can also `cat report.md` file which appeared in the project directory and contains the "report card" of the run, i.e. a bunch of evaluations and metrics. At the very end, you'll see a summary table, for example:

---

- Characters: 333,989
- Lines: 8,304
- Files: 44
- Tokens (approx): 83,497
- Dependencies (uv.lock lines): 2,004

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2219   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |
| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval       | -        | 0.0671   | 0.0854   | -        |
| MMLU            | -        | 0.3111   | 0.3151   | -        |
| ChatCORE        | -        | 0.0730   | 0.0884   | -        |

Total wall clock time: 3h51m

---

(Your table might be missing the RL number by default). For a lot more information around the speedrun script and what to look for and expect, please refer to the walkthrough that I posted in Discussions of the repo: ["Introducing nanochat: The best ChatGPT that $100 can buy"](https://github.com/karpathy/nanochat/discussions/1).

## Bigger models

Unsurprisingly, $100 is not enough to train a highly performant ChatGPT clone. In fact, LLMs are famous for their multi-million dollar capex. For our purposes, I think there are two more scales of interest. First is the ~$300 tier d26 model (i.e. depth=26) that trains in ~12 hours, which slightly outperforms GPT-2 CORE score. Second is the $1000 tier (~41.6 hours), just because it's a nice round number. But both of these are not yet fully supported and therefore not attached here in the master branch yet.

That said, to give a sense, the example changes needed for the [speedrun.sh](speedrun.sh) file to train a GPT-2 grade model d26 only involve three changes:

```bash
...
# you'll need to download more data shards for pretraining
# get the number of parameters, multiply 20 to get tokens, multiply by 4.8 to get chars,
# divide by 250 million to get number of shards. todo need to improve this...
python -m nanochat.dataset -n 450 &
...
# use --depth to increase model size. to not oom, halve device batch size 32 -> 16:
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
...
# make sure to use the same later during midtraining:
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

That's it! The biggest thing to pay attention to is making sure you have enough data shards to train on (the code will loop and do more epochs over the same training set otherwise, decreasing learning speed a bit), and managing your memory/VRAM, primarily by decreasing the `device_batch_size` until things fit (the scripts automatically compensate by increasing the number of gradient accumulation loops, simply turning parallel compute to sequential compute).

And a bit more about computing environments that will run nanochat:

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.
- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device_batch_size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't implemented this out of the box so it might take a bit of tinkering.

## Running on CPU / MPS

nanochat can be run on CPU or on MPS (if you're on Macbook), and will automatically try to detect what device is best to run on. You're not going to get too far without GPUs, but at least you'll be able to run the code paths and maybe train a tiny LLM with some patience. For an example of how to make all the run commands much smaller (feel free to tune!), you can refer to [dev/runcpu.sh](dev/runcpu.sh) file. You'll see that I'm essentially restricting all scripts to train smaller models, to run for shorter number of iterations, etc. This functionality is new, slightly gnarly (touched a lot of code), and was merged in this [CPU|MPS PR](https://github.com/karpathy/nanochat/pull/88) on Oct 21, 2025.

## Customization

To customize your nanochat, see [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139) in Discussions, which describes how you can tune your nanochat's personality through synthetic data generation and mixing that data into midtraining and SFT stages.

Additionally, to add new abilities to nanochat, see [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164).

## Questions

nanochat is designed to be short and sweet. One big advantage of this is that we can package up all of the files together and copy paste them to your favorite LLM to ask arbitrary questions. As an example, I like to package up the repo using the [files-to-prompt](https://github.com/simonw/files-to-prompt) utility like so:

```bash
files-to-prompt . -e py -e md -e html -e toml -e sh --cxml > packaged.txt
```

This includes all py, html, toml, sh files and chooses the cxml output format. Everything is written to the `packaged.txt` file, which atm measures ~330KB (i.e. well below ~100K tokens for a state of the art LLM), and ~8K lines of code in 45 files.

Alternatively, I recommend using [DeepWiki](https://deepwiki.com/karpathy/nanochat) from Devin/Cognition to ask questions of this repo. In the URL of this repo, simply change github.com to deepwiki.com, and you're off.

## Tests

I haven't invested too much here but some tests exist, especially for the tokenizer. Run e.g. as:

```bash
python -m pytest tests/test_engine.py -v -s
```

## File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # Example synthetic data for identity
│   ├── generate_logo.html
│   ├── nanochat.png
│   ├── repackage_data_reference.py # Pretraining data shard generation
│   └── runcpu.sh                   # Small example of how to run on CPU/MPS
├── nanochat
│   ├── __init__.py                 # empty
│   ├── adamw.py                    # Distributed AdamW optimizer
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Misc small utilities, quality of life
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing Distributed Data Loader
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV Cache
│   ├── execution.py                # Allows the LLM to execute Python code as tool
│   ├── gpt.py                      # The GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── muon.py                     # Distributed Muon optimizer
│   ├── report.py                   # Utilities for writing the nanochat Report
│   ├── tokenizer.py                # BPE Tokenizer wrapper in style of GPT-4
│   └── ui.html                     # HTML/CSS/JS for nanochat frontend
├── pyproject.toml
├── run1000.sh                      # Train the ~$800 nanochat d32
├── scripts
│   ├── base_eval.py                # Base model: calculate CORE score
│   ├── base_loss.py                # Base model: calculate bits per byte, sample
│   ├── base_train.py               # Base model: train
│   ├── chat_cli.py                 # Chat model (SFT/Mid): talk to over CLI
│   ├── chat_eval.py                # Chat model (SFT/Mid): eval tasks
│   ├── chat_rl.py                  # Chat model (SFT/Mid): reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   ├── chat_web.py                 # Chat model (SFT/Mid): talk to over WebUI
│   ├── mid_train.py                # Chat model: midtraining
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── speedrun.sh                     # Train the ~$100 nanochat d20
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # Make Task from arbitrary jsonl convos
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   ├── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
│   └── spellingbee.py              # Task teaching model to spell/count letters
├── tests
│   └── test_engine.py
└── uv.lock
```

## Contributing

nanochat is nowhere near finished. The goal is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there will be no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a concrete ChatGPT clone and its report card.

Current LLM policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

---

## C++ Code Model Training (Fork Extensions)

This fork extends nanochat for **C++ code generation** with custom architectures and TPU training.

### Current Training Run

**Machine**: Google Cloud TPU v6e-8 (8 Trillium chips, 31.25 GB HBM each)

| Parameter | Value |
|-----------|-------|
| **Model** | d=24, 877M params, AAM hybrid (Attention-Attention-Mamba) |
| **Context** | 65,536 tokens (64K) |
| **Parallelism** | TP=4 (tensor), DP=2 (data) via XLA SPMD 2D mesh |
| **Batch** | 524K tokens/step, 4 grad accum steps |
| **Training tokens** | 26.2B (30:1 token:param ratio, 50K iterations) |
| **Throughput** | ~200K tok/sec, ~2.5s/step |
| **ETA** | ~36 hours |

**Features**: Mamba-3 (QK-norm, bias, complex RoPE, trapezoidal), Engram (layers 0,3,6),
mHC, DSA (layers 7-23), MTP (lambda=0.3), FIM (50%), gradient checkpointing, XLA flash attention.

### Training Data

Dataset: **cpp_compilable_64k** — compilable-ordered C++ training documents.

| Dataset | Docs | Size | Tokens | Description |
|---------|------|------|--------|-------------|
| cpp_crossfile_16k | 8.1M | 23 GB | 16K | Tree-sitter cross-file (JSONL) |
| cpp_project_crossfile_16k | 1.0M | 8.0 GB | 16K | Project-aware cross-file |
| cpp_clang_crossfile_16k | 794K | 883 MB | 16K | Clang semantic indexer |
| cpp_compilable_16k | 1.0M | 7.3 GB | 16K | Compilable: types->functions |
| **cpp_compilable_64k** | **394K** | **3.4 GB** | **64K** | **Compilable: types->functions (active)** |

All datasets at `gs://nanochat-training-data-2026/data/`. Generated from 93 C++ open-source
projects (641K files, 29 GB raw) using `tools/cpp_chunker` (Rust).

**Compilable ordering**: Each document is structured as a near-compilable C++ translation unit:
1. Preamble: `#include`, `using`, forward declarations
2. Type definitions: structs/classes/enums in topological order (Base before Derived)
3. Functions: bottom-up order (leaf callees first, callers last)

### Architecture Extensions

| Component | Description |
|-----------|-------------|
| **Mamba-3 Hybrid** | SSM layers interleaved with attention (AAM pattern). Adds QK-norm, learnable B/C bias, complex RoPE, and trapezoidal discretization to Mamba-2. |
| **Engram** | N-gram prediction branches at early layers for local pattern learning. |
| **mHC** | Multi-Head Collaboration via Sinkhorn transport for cross-head information mixing. |
| **DSA** | DeepSeek Sparse Attention — top-k token selection for O(n*k) attention in upper layers. |
| **MTP** | Multi-Token Prediction (DeepSeek-V3 style) — predicts token i+2 as auxiliary loss. |
| **FIM** | Fill-in-the-Middle — 50% of documents randomly rearranged for infilling capability. |

### Tensor Parallelism

Model weights are split across TPU chips using Megatron-style SPMD sharding.
`num_heads` must be divisible by `tp_degree`:

| Depth | Params | Heads | TP=2 | TP=4 | TP=8 |
|-------|--------|-------|------|------|------|
| 16 | 270M | 8 | OK | OK | OK |
| **24** | **877M** | **12** | OK | **OK** | NO |
| 32 | 1.5B | 16 | OK | OK | OK |
| 48 | 3.4B | 24 | OK | OK | OK |

See [docs/TENSOR_PARALLELISM.md](docs/TENSOR_PARALLELISM.md) for full details.

### Launch Command

```bash
source ~/venv311/bin/activate && source ~/.tpu_env
cd ~/nanochat && export NANOCHAT_BASE_DIR=/home/dave/data

python3 -u -m scripts.base_train \
    --depth=24 --num_iterations=50000 \
    --tensor_parallel=4 \
    --device_batch_size=1 --max_seq_len=65536 --total_batch_size=524288 \
    --kernel=current --no_compile --xla_flash_attn \
    --window_pattern=L \
    --mamba --mamba_pattern=AAM \
    --mamba3_qknorm --mamba3_bias --mamba3_complex_rope --mamba3_trapezoidal \
    --engram --engram_layers=0,3,6 \
    --mhc --mtp --mtp_lambda=0.3 \
    --dsa --dsa_start_layer=7 \
    --gradient_checkpointing \
    --fim_rate=0.5 \
    --data_dir=/home/dave/data/parquet --streaming_data \
    --run=dummy \
    --core_metric_every=5000 --save_every=5000 --sample_every=5000
```

### Documentation

- [CLAUDE.md](CLAUDE.md) — Full development guide (GPU benchmarks, TPU setup, data pipeline)
- [docs/TPU_SETUP.md](docs/TPU_SETUP.md) — TPU environment setup and version matrix
- [docs/TENSOR_PARALLELISM.md](docs/TENSOR_PARALLELISM.md) — TP sharding details and dimension tables
- [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) — C++ data processing pipeline

## License

MIT
