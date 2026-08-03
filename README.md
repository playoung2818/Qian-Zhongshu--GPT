# Reanimation Jutsu Workflow

This project creates style adapters for Qwen2.5 and serves them through one Hugging Face Space.

The current Space provides two assistants:

- Qian Zhongshu style assistant
- Abraham Lincoln inspired historical assistant

Both assistants share `Qwen/Qwen2.5-7B-Instruct`. Each assistant uses a separate LoRA adapter.

## Deployed resources

| Resource | Hugging Face repository | Access |
|---|---|---|
| Chat application | `Playoung2818/Zhongshu_Qian` | Space |
| Lincoln adapter | `Playoung2818/lincoln-qwen2.5-7b-lora` | Private model |
| Lincoln dataset | `Playoung2818/lincoln-style-chat-data` | Private dataset |
| Base model | `Qwen/Qwen2.5-7B-Instruct` | Public model |

Open the [Reanimation Jutsu Space](https://huggingface.co/spaces/Playoung2818/Zhongshu_Qian).

## Workflow

```text
[Source documents](source_data/lincoln_speeches_letters.txt)
      |
      v
Dataset preparation script
      |
      +--> Historical topic-response examples
      +--> Historical continuation examples
      +--> Reviewed modern-domain examples
      |
      v
Train and validation JSONL files
      |
      v
Private Hugging Face dataset
      |
      v
L4 Hugging Face Job
      |
      v
Lincoln LoRA adapter
      |
      v
Qwen base model + Qian adapter + Lincoln adapter
      |
      v
Two-tab Gradio Space
```

## Important files

| File | Purpose |
|---|---|
| `source_data/lincoln_speeches_letters.txt` | Stores the public-domain Lincoln source text. |
| `scripts/prepare_lincoln_dataset.py` | Creates historical train and validation examples. |
| `example_data/lincoln_synthetic_modern.jsonl` | Stores reviewed modern-domain examples. |
| `example_data/lincoln_train.jsonl` | Stores the final train split. |
| `example_data/lincoln_eval.jsonl` | Stores the validation split. |
| `scripts/train_lincoln_hf_job.py` | Runs QLoRA on Hugging Face Jobs. |
| `huggingface_space/app.py` | Loads both adapters and creates both chat tabs. |
| `huggingface_space/requirements.txt` | Defines the Space dependencies. |
| `huggingface_space/README.md` | Defines the Space metadata. |

## Data design

The Lincoln source comes from Project Gutenberg eBook 14721.

The preparation script extracts documents, splits passages, and creates two historical task types.

The `topic_response` task uses a document title as the prompt. Lincoln's text becomes the target response.

The `continuation` task uses a passage opening as the prompt. The remaining passage becomes the target response.

The modern-domain file contains reviewed responses about current issues. These examples separate historical evidence from reasoned speculation.

The final train split contains 364 historical examples and 16 modern-domain examples. The validation split contains 40 historical examples.

## 1. Prepare the Lincoln dataset

Install the required Python packages.

```bash
pip install -r requirements_qwen.txt
```

Place the Project Gutenberg text at this path.

```text
source_data/lincoln_speeches_letters.txt
```

Create the train and validation files.

```bash
python scripts/prepare_lincoln_dataset.py
```

Verify the row counts.

```bash
wc -l example_data/lincoln_train.jsonl example_data/lincoln_eval.jsonl
```

The expected counts are 380 train rows and 40 validation rows.

## 2. Upload the Lincoln dataset

Authenticate with Hugging Face.

```bash
hf auth login
```

Create the private dataset repository once.

```bash
hf repos create Playoung2818/lincoln-style-chat-data --type dataset --private --exist-ok
```

Upload the train split.

```bash
hf upload Playoung2818/lincoln-style-chat-data \
  example_data/lincoln_train.jsonl train.jsonl \
  --type dataset
```

Upload the validation split.

```bash
hf upload Playoung2818/lincoln-style-chat-data \
  example_data/lincoln_eval.jsonl validation.jsonl \
  --type dataset
```

## 3. Train the Lincoln LoRA adapter

Set `HF_TOKEN` in the local shell. Use a token with write access.

```bash
export HF_TOKEN="YOUR_WRITE_TOKEN"
```

Start the L4 job.

```bash
hf jobs uv run scripts/train_lincoln_hf_job.py \
  --flavor l4x1 \
  --secrets HF_TOKEN \
  --timeout 2h
```

List the jobs.

```bash
hf jobs ps --all
```

Read the job log.

```bash
hf jobs logs JOB_ID
```

The job saves the adapter to `Playoung2818/lincoln-qwen2.5-7b-lora`.

The adapter repository must contain these files:

- `adapter_config.json`
- `adapter_model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`

## 4. Configure the Space

Open the Space settings page.

Add `HF_TOKEN` under **Variables and secrets**.

Use a token that can read the private Lincoln adapter.

Select ZeroGPU or paid GPU hardware.

The current application supports ZeroGPU through `@spaces.GPU`.

## 5. Deploy both chat tabs

Verify the application syntax.

```bash
python -m py_compile huggingface_space/app.py
```

Upload the Space files.

```bash
hf upload Playoung2818/Zhongshu_Qian huggingface_space \
  --type space \
  --exclude '__pycache__/*' \
  --commit-message "Update Reanimation Jutsu"
```

Verify the Space status.

```bash
hf spaces info Playoung2818/Zhongshu_Qian \
  --expand runtime,sha,lastModified \
  --format json
```

Wait until the runtime stage reports `RUNNING`.

## Runtime behavior

The Space starts before it loads Qwen. This design lets Hugging Face detect the ZeroGPU functions.

The first chat request loads the quantized Qwen base model. It also loads both named LoRA adapters.

The Qian tab selects the `qian` adapter. The Lincoln tab selects the `lincoln` adapter.

The queue permits one request at a time. This rule prevents an adapter change during another response.

## Evaluate the adapters

Use prompts that do not occur in the train split.

Score each answer against these criteria:

- Style fidelity
- Factual accuracy
- Prompt relevance
- Readability
- Quote integrity

For Lincoln responses, verify every quotation against a primary source.

For modern topics, verify that the answer labels inference as inference.

## Troubleshooting

### `RepositoryNotFoundError`

Verify the adapter repository identifier.

Verify that the `HF_TOKEN` secret exists in the Space settings.

Verify that the token can read the private model repository.

### `No CUDA GPUs are available`

Keep all model and adapter loads inside a function with `@spaces.GPU`.

Do not load the model during application startup on ZeroGPU.

### `No @spaces.GPU function detected`

Keep the decorated response functions at module scope.

Start Gradio without a long model load at module scope.

### A response uses the wrong style

Verify that `model.set_adapter()` receives the correct adapter name.

Keep the queue concurrency limit at one.

## Project origin

This workspace started from [Suffoquer-fang/LuXun-GPT](https://github.com/Suffoquer-fang/LuXun-GPT).

The current workflow uses Qwen2.5, PEFT, TRL, Hugging Face Jobs, and Gradio.

## Responsible use

The Lincoln assistant does not claim to be Abraham Lincoln.

The assistant must not invent quotations or present modern opinions as historical facts.

The Qian assistant produces style imitation. Verify the applicable copyright rules before public distribution.
