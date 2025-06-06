#!/bin/bash
pip install torch torchao torchtune
python -c 'import torchtune; print("Torchtune version:", torchtune.__version__)'

tune download \
  meta-llama/Llama-2-7b-hf \
  --output-dir /usr/local/torchtune/checkpoints \
  --hf-token $HF_TOKEN
