FROM python:3.11-slim
WORKDIR /
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt update
RUN apt install -y --no-install-recommends  libgl1 libsm6  libxrender1  libfontconfig1 libglib2.0-0
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=8000"]