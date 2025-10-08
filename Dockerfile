# RLDK Demo Dockerfile
# Lightweight container for running RLDK demo

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy source code first (required for editable install)
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p test_artifacts/logs_clean \
    test_artifacts/logs_doctored_kl_spike \
    test_artifacts/reward_drift_demo \
    test_artifacts/ckpt_identical \
    test_artifacts/ckpt_value_head_edit \
    rldk_reports

# Generate test fixtures
RUN python3 tests/_make_fixtures.py

# Generate training logs
RUN python3 generate_logs.py

# Make demo script executable
RUN chmod +x scripts/demo.sh

# Expose port for any web interfaces (if needed)
EXPOSE 8080

# Set the default command to run the demo
CMD ["scripts/demo.sh"]