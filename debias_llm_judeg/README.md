## Environment setup

We use **uv** as the primary dependency manager, but you can also use `conda` or any other environment tool.

### Option 1: Recommended (uv)

```
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and sync the env
uv venv
uv sync

# Run commands [Example]
uv run src/train_judge.py
```

### Option 2: conda

```
conda create -n llm-judge python=3.11
conda activate llm-judge
pip install -r requirements.txt

```

