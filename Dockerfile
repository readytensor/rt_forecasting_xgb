# Use an NVIDIA CUDA base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 as builder

# Install OS dependencies and cleanup in a single RUN to reduce layers and ensure minimal size
RUN apt-get update && apt-get install -y --no-install-recommends \
         ca-certificates \
         dos2unix \
         python3.9 \
         python3-pip \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && ln -sf /usr/bin/python3.9 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.9 -m pip install --upgrade pip

# copy requirements file and and install
COPY ./requirements.txt /opt/
RUN python3.9 -m pip install --no-cache-dir -r /opt/requirements.txt

# Copy source code and scripts into image and set permissions
COPY src /opt/src
COPY ./entry_point.sh /opt/
COPY ./fix_line_endings.sh /opt/
RUN chmod +x /opt/entry_point.sh \
    && chmod +x /opt/fix_line_endings.sh \
    && /opt/fix_line_endings.sh "/opt/src" \
    && /opt/fix_line_endings.sh "/opt/entry_point.sh"

# Set working directory
WORKDIR /opt/src

# Set Python environment variables
ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    PATH="/opt/src:${PATH}"

# Set non-root user
USER 1000

# Set entrypoint
ENTRYPOINT ["/opt/entry_point.sh"]
