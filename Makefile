serve:
	mlx_lm.server \
		--host 127.0.0.1 \
		--port 13333 \
		--model Qwen/Qwen3-8B \
		--draft-model Qwen/Qwen3-0.6B \
		--max-tokens 64000

run:
	chainlit run -w main.py

install:
	uv venv --seed -p 3.12 .venv && uv pip install -r pyproject.toml
