FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
ENV UVICORN_WS_PROTOCOL=websockets
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
RUN uv sync --frozen
EXPOSE 7860
CMD ["uv", "run", "chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]