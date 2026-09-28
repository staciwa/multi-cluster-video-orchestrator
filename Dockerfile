FROM minizinc/minizinc:2.9.7-jammy

# Set working directory
WORKDIR /workspace

USER root

# Install python and minimal necessary tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv curl jq \
    && rm -rf /var/lib/apt/lists/*

# Install 'uv' by copying the binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Pre-install dependencies to take advantage of Docker layer caching
# This assumes pyproject.toml and uv.lock are in the build context root
COPY pyproject.toml uv.lock* .python-version ./
RUN uv sync --frozen --no-dev --python 3.12

# Copy the rest of the application code
COPY . .

# Set entrypoint to bash to allow flexible script execution
ENTRYPOINT ["/bin/bash"]

# Default: interactive shell (run pipeline steps manually, see README.md)
CMD ["-c", "echo 'Container ready. Run pipeline steps from README.md, e.g.:' && echo '  uv run python minizinc/generate_reference_dataset.py --count 100 --output-dir dataset_raw' && exec bash"]
