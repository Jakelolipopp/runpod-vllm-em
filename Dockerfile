FROM nvidia/cuda:12.9.1-base-ubuntu22.04 

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install base utilities (Removed python3-pip to avoid system conflicts)
RUN apt-get update -y \
    && apt-get install -y wget curl git dos2unix \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Miniconda for an isolated, guaranteed Python environment
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# 3. Add Conda to the system PATH
ENV PATH="/opt/conda/bin:$PATH"

RUN ldconfig /usr/local/cuda-12.9/compat/

# 4. Install vLLM with FlashInfer using Conda's Python
# (Note: Miniconda uses 'python' rather than 'python3')
RUN python -m pip install --upgrade pip && \
    python -m pip install "vllm[flashinfer]==0.19.0" --extra-index-url https://download.pytorch.org/whl/cu129

# Install additional Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade -r /requirements.txt

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME="surogate/Qwen3.5-0.8B-NVFP4"
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/models"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""
ARG VLLM_NIGHTLY="false"

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4

ENV PYTHONPATH="/:/vllm-workspace"

RUN if [ "${VLLM_NIGHTLY}" = "true" ]; then \
    pip install -U vllm --pre --index-url https://pypi.org/simple --extra-index-url https://wheels.vllm.ai/nightly && \
    pip install git+https://github.com/huggingface/transformers.git; \
fi

COPY src /src

# Strip Windows line endings from all scripts
RUN find /src -type f \( -name "*.py" -o -name "*.sh" \) -exec dos2unix {} +

RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python /src/download_model.py; \
    fi

# Start the handler using the absolute path to the Conda Python executable
CMD ["/opt/conda/bin/python", "-u", "/src/handler.py"]
