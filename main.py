"""
Phase 4 demo — PyTorch MNIST training.

환경변수:
  EPOCHS              : 학습 epoch 수 (default: 3)
  AIMIMO_INPUT_DIR    : 입력 데이터 dir (Init Container 가 mc cp 한 위치, default: /workspace/input)
                        구조: {INPUT_DIR}/MNIST/raw/train-images-idx3-ubyte.gz 등
  AIMIMO_OUTPUT_DIR   : 결과 저장 dir (Sidecar / inline 이 mc cp 로 upload, default: /workspace/output)
  AIMIMO_USER         : 실행자 (logging 용, optional)
  AIMIMO_EXPERIMENT_ID: 실험 ID (logging 용, optional)
"""

import gzip
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def prepare_data(input_dir: str) -> None:
    """MinIO 에서 받은 .gz 파일을 압축 해제.

    torchvision 의 datasets.MNIST(download=False) 는 raw folder 안에 압축 해제된 binary
    (`train-images-idx3-ubyte` 등) 만 인식. .gz 파일을 곁에 압축 해제해 둔다.
    """
    raw_dir = os.path.join(input_dir, "MNIST", "raw")
    if not os.path.isdir(raw_dir):
        raise SystemExit(f"raw dir 없음: {raw_dir} — Init Container 의 mc cp 결과 확인")

    extracted = 0
    for fn in sorted(os.listdir(raw_dir)):
        if not fn.endswith(".gz"):
            continue
        src = os.path.join(raw_dir, fn)
        dst = src[:-3]  # .gz 제거
        if os.path.exists(dst):
            continue
        with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        extracted += 1
    if extracted:
        print(f"  Extracted {extracted} .gz files in {raw_dir}")


def main() -> None:
    epochs = int(os.environ.get("EPOCHS", 3))
    input_dir = os.environ.get("AIMIMO_INPUT_DIR", "/workspace/input")
    output_dir = os.environ.get("AIMIMO_OUTPUT_DIR", "/workspace/output")
    user = os.environ.get("AIMIMO_USER", "unknown")
    exp_id = os.environ.get("AIMIMO_EXPERIMENT_ID", "local")

    print("=" * 60)
    print(f"  aimimo demo — MNIST training")
    print(f"  user      : {user}")
    print(f"  experiment: {exp_id}")
    print(f"  input dir : {input_dir}")
    print(f"  output dir: {output_dir}")
    print(f"  epochs    : {epochs}")
    print("=" * 60)

    # GPU 확인
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — GPU 필요")
    device = torch.device("cuda")
    print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"  CUDA      : {torch.version.cuda}")
    print(f"  PyTorch   : {torch.__version__}")
    print()

    # 데이터 준비 — .gz → binary 압축 해제 (download=False 이므로 직접 처리)
    prepare_data(input_dir)

    # 데이터 로드
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(input_dir, train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    print(f"  Train samples: {len(train_dataset)}  ({len(train_loader)} batches)\n")

    # 작은 CNN
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3)
            self.conv2 = nn.Conv2d(32, 64, 3)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Training
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(f"  Epoch {epoch}/{epochs}  batch {i:4d}/{len(train_loader)}  loss {loss.item():.4f}")
    elapsed = time.time() - start
    print(f"\n  Training done in {elapsed:.1f}s")

    # 결과 저장 (Sidecar / inline 이 MinIO 로 upload)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "mnist_cnn.pt")
    torch.save(model.state_dict(), out_path)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved → {out_path}  ({size_mb:.1f} MB)")

    # 메타데이터 — 결과 reproducibility
    meta_path = os.path.join(output_dir, "meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"user={user}\n")
        f.write(f"experiment_id={exp_id}\n")
        f.write(f"epochs={epochs}\n")
        f.write(f"gpu={torch.cuda.get_device_name(0)}\n")
        f.write(f"elapsed_seconds={elapsed:.1f}\n")
    print(f"  Saved → {meta_path}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
