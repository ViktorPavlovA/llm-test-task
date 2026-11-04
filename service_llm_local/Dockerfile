FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ARG LLAMA_CPP_TAG=v0.8.0
ARG CUDA_ARCHITECTURES=

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMA_CPP_TAG}" \
        https://github.com/ggml-org/llama.cpp.git llama.cpp

WORKDIR /src/llama.cpp
RUN set -eu; \
    extra=""; \
    if [ -n "${CUDA_ARCHITECTURES}" ]; then \
        extra="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"; \
    fi; \
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DGGML_BACKEND_DL=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_CURL=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
        ${extra} \
    && cmake --build build --config Release --target llama-server \
    && mkdir -p /opt/llama.cpp/bin \
    && cp -a build/bin/llama-server /opt/llama.cpp/bin/ \
    && find build -type f -name '*.so*' -exec cp -a {} /opt/llama.cpp/bin/ \; \
    && test -x /opt/llama.cpp/bin/llama-server

RUN set -eu; \
    cp -a /src/llama.cpp/build/bin/. /opt/llama.cpp/bin/; \
    cd /opt/llama.cpp/bin; \
    for lib in *.so.*; do \
        [ -f "$lib" ] || continue; \
        soname=$(printf '%s\n' "$lib" | sed -E 's/^(.*\.so\.[0-9]+).*/\1/'); \
        if [ "$soname" != "$lib" ] && [ ! -e "$soname" ]; then \
            ln -s "$lib" "$soname"; \
        fi; \
    done; \
    test -e libllama-common.so.0; \
    test -e libllama.so.0; \
    test -e libggml.so.0; \
    test -e libggml-base.so.0; \
    test -e libmtmd.so.0

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/llama.cpp/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/llama.cpp/bin:/usr/local/cuda/lib64 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8686 \
    MODEL_CONFIG=/models/config.yaml

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        python3 \
        python3-yaml \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app /models

COPY --from=builder /opt/llama.cpp/bin /opt/llama.cpp/bin
COPY main.py /app/main.py
COPY models/config.yaml /models/config.yaml
COPY models/download_model_script.py /models/download_model_script.py


RUN set -eu; \
    cd /opt/llama.cpp/bin; \
    if ldd ./llama-server | grep -E 'lib(llama|ggml|mtmd).*(not found)'; then \
        echo "llama-server is missing shared libraries"; \
        ldd ./llama-server; \
        exit 1; \
    fi; \
    if ldd ./libggml-cuda.so | grep 'not found' | grep -v 'libcuda.so.1'; then \
        echo "CUDA backend is missing a library other than the host driver"; \
        ldd ./libggml-cuda.so; \
        exit 1; \
    fi

WORKDIR /app
EXPOSE 8686
VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD curl -sf http://127.0.0.1:8686/health || exit 1

CMD ["python3", "/app/main.py"]
