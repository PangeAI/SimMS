FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /code
COPY ./pyproject.toml /code/pyproject.toml

ENV PYTHONUNBUFFERED=1 \
	GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_THEME=huggingface \
	SYSTEM=spaces

COPY simms/ /code/simms/
COPY tests/ /code/tests/
COPY README.md /code/README.md
RUN pip install --no-cache-dir --upgrade -e .[dev]
COPY app.py /code/app.py
RUN pip install --no-cache-dir --upgrade gradio
# Copy the current directory contents into the container at $HOME/app setting the owner to the user
CMD ["python3", "app.py"]