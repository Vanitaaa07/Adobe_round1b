# --------------------- BUILDER STAGE ---------------------
FROM python:3.9-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# Install heavy packages with retries first
RUN pip install --default-timeout=200 --retries=15 gensim fasttext spacy

# Install remaining requirements
RUN pip install --default-timeout=100 --retries=10 --no-cache-dir -r requirements.txt

# Install spaCy model and link it so it's globally recognized
RUN python -m spacy download en_core_web_sm \
    && python -m spacy link en_core_web_sm en_core_web_sm

# Download NLTK resources
ENV NLTK_DATA=/usr/local/nltk_data
RUN mkdir -p ${NLTK_DATA} \
    && python -c "import nltk; nltk.download('punkt', download_dir='${NLTK_DATA}'); \
                  nltk.download('stopwords', download_dir='${NLTK_DATA}'); \
                  nltk.download('punkt_tab', download_dir='${NLTK_DATA}')"

# --------------------- RUNTIME STAGE ---------------------
FROM python:3.9-slim-bookworm AS runner

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages and model/data from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/nltk_data /usr/local/nltk_data

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV NLTK_DATA=/usr/local/nltk_data

# Default run command
CMD ["python","-u" , "main.py"]
