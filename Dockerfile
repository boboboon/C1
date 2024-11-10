# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project files
COPY . /app

# Install necessary system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git && \
    # Upgrade pip and install dependencies from pyproject.toml
    pip install --upgrade pip setuptools wheel && \
    pip install . && \
    # Clean up unnecessary files
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Expose any ports the app might use (e.g., 8888 for Jupyter Notebook)
EXPOSE 8888

# Run the main command by default
CMD ["bash"]


# docker build -t gbm .
# docker run -it gbm     