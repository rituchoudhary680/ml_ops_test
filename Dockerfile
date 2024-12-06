FROM python:3.9-slim

WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy project files
COPY . /app/

RUN pip install -r requirements.txt

# Install dependencies
RUN poetry install

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["poetry", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
