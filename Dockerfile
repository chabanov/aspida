# Aspida — reproducible Linux build of the encrypted inference server.
#
# Builds the Ada/SPARK engine + secure_server on Ubuntu using Alire's GNAT
# toolchain. Models (GGUF) are NOT baked in — mount them at runtime and point
# QWEN_MODEL_PATH at the file.
#
#   docker build -t aspida-server .
#   docker run --rm -p 8765:8765 \
#       -e QWEN_MODEL_PATH=/models/model.gguf \
#       -v /host/models:/models:ro aspida-server
#
# Note: this image is CPU-only. A GPU image (CUDA/ROCm base) comes with Stream B.

FROM ubuntu:24.04 AS build
ARG ALR_VERSION=2.0.2
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl unzip pkg-config \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/alr.zip \
      "https://github.com/alire-project/alire/releases/download/v${ALR_VERSION}/alr-${ALR_VERSION}-bin-x86_64-linux.zip" \
 && unzip -q /tmp/alr.zip -d /tmp && install -m755 /tmp/bin/alr /usr/local/bin/alr \
 && rm -rf /tmp/alr.zip /tmp/bin
RUN alr toolchain --select gnat_native gprbuild
WORKDIR /src
COPY . .
# ARCH=portable keeps the binary runnable on any x86-64 (no -march=native); use
# ARCH=native when the build host == the run host for a small speedup.
RUN make server ARCH=portable

FROM ubuntu:24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/obj/secure_server /app/secure_server
COPY --from=build /src/obj/secure_client /app/secure_client
EXPOSE 8765
ENTRYPOINT ["/app/secure_server"]
CMD ["8765"]
