# PyTorch + CUDA base image (Docker Hub, ~3GB).
# 이전 nvcr.io/nvidia/pytorch:24.10-py3 (12GB) 대체:
#   - foreign layer 없어서 in-cluster registry 로 push 시 nvcr.io 재호출 안 함
#   - 1/4 크기라 cold cache 첫 빌드 5-7분으로 단축
#
# 포함: PyTorch 2.4.1 + torchvision + CUDA 12.1 + cuDNN 9 + Python 3.x
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY main.py .

# 학습 진행 print 가 사용자에게 즉시 보이게
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
