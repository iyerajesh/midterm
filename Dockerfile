FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
USER root
RUN apt-get update && apt-get install -y \
    gcc g++ cmake make \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    protobuf-compiler \
    libprotobuf-dev \
    libssl-dev \
    libffi-dev \
    patchelf 
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
ENV UVICORN_WS_PROTOCOL=websockets
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
RUN uv sync --frozen || True
RUN uv pip install cython
RUN uv pip install --upgrade langchain langchain-core langchain-openai langchain-huggingface
EXPOSE 7860
CMD ["uv", "run", "chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
