# PyTorch + CUDA base image (NVIDIA NGC).
# 약 7-8GB — 첫 build 시 시간 걸림. 두 번째부터 layer cache.
FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /app

COPY main.py .

# 학습 진행 print 가 사용자에게 즉시 보이게
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
