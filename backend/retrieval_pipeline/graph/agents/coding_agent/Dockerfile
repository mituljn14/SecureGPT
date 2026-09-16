# Dockerfile
#
# Sandbox image for the Coding Agent's Step 5 (Docker Python Sandbox).
# Build once:
#   docker build -t coding-agent-sandbox -f Dockerfile .
#
# This image only contains the libraries the generated code is allowed to use.
# No network tools, no extra utilities.

FROM python:3.11-slim

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    openpyxl \
    pymupdf \
    python-docx

# Non-root user — generated code never runs as root inside the container.
RUN useradd -m -u 10001 sandboxuser
USER sandboxuser

WORKDIR /workspace
