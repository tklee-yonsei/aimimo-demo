# aimimo demo — PyTorch MNIST training

Phase 4 의 build → run 흐름 시연용 minimal sample.

> 이 디렉토리는 **별도 public GitHub repo 로 옮겨서 사용** 권장. Kaniko 가 그 repo 의 git context 로 build.

## 흐름

```
[사용자: git push to public repo]
        ↓
[aimimo-run CLI]
  ├─ Build Job (Kaniko, F 노드)
  │    git clone (commit 기준) + Dockerfile build
  │    → push to registry.registry.svc:5000/<user>/demo-mnist:<commit>
  └─ Run Job (h100 노드)
       ├─ Init: mc cp lab/user-tklee/mnist/ → /workspace/input/MNIST/
       ├─ Main: python main.py (5 epoch ~1-2분)
       └─ inline: mc cp /workspace/output/ → lab/user-tklee/experiments/.../
```

## 파일

| 파일 | 역할 |
|---|---|
| Dockerfile | nvcr.io/nvidia/pytorch:24.10-py3 base + main.py copy |
| main.py | MNIST CNN training, env 변수로 EPOCHS 받음 |

## 환경변수

| name | 용도 | default |
|---|---|---|
| `EPOCHS` | 학습 epoch 수 | 3 |
| `AIMIMO_INPUT_DIR` | MNIST 데이터 위치 (Init Container 가 fetch 한 곳) | `/workspace/input` |
| `AIMIMO_OUTPUT_DIR` | model + meta 저장 위치 (Sidecar / inline 이 upload) | `/workspace/output` |
| `AIMIMO_USER` | logging | unknown |
| `AIMIMO_EXPERIMENT_ID` | logging | local |

## 사전 작업 (admin, 1회)

### 1. MNIST 데이터를 MinIO 에 업로드 (사용자 본인 영역)

```bash
mc alias set lab https://165.132.192.75:32700 <access-key> <secret-key>

# 임시 dir 에 download
mkdir -p /tmp/mnist/MNIST/raw && cd /tmp/mnist/MNIST/raw
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz  t10k-labels-idx1-ubyte.gz; do
  wget "https://ossci-datasets.s3.amazonaws.com/mnist/$f"
done

# MinIO 업로드 — 본인 영역 (user-tklee)
mc cp --recursive /tmp/mnist/ lab/user-tklee/mnist/

# 검증
mc ls --recursive lab/user-tklee/mnist/
# → user-tklee/mnist/MNIST/raw/train-images-idx3-ubyte.gz 등 4 파일
```

### 2. K8s Secret 등록 (Run Job 의 Init Container 가 사용)

```bash
ssh dclserver75 "kubectl create secret generic minio-tklee -n experiments \
  --from-literal=access-key='<your-access-key>' \
  --from-literal=secret-key='<your-secret-key>'"

# 검증
ssh dclserver75 "kubectl get secret minio-tklee -n experiments -o jsonpath='{.data.access-key}' | base64 -d"
```

### 3. 별도 public repo 생성 + 코드 push

```bash
# GitHub 에서 새 public repo 생성 (예: aimimo-demo)
# 그 후
git clone https://github.com/<your-user>/aimimo-demo.git
cd aimimo-demo
cp -r /path/to/infra/phase4/demo/* .
git add .
git commit -m "init"
git push
```

## 시연 — aimimo-run CLI 실행

```bash
COMMIT=$(git rev-parse HEAD)   # 또는 짧은 SHA

./infra/phase4/cli/aimimo-run \
  --user tklee \
  --project demo-mnist \
  --repo <your-user>/aimimo-demo \
  --branch main \
  --commit $COMMIT \
  --dockerfile Dockerfile \
  --workdir /app \
  --command 'python main.py' \
  --env EPOCHS=3 \
  --dataset 'user-tklee/mnist/=>/workspace/input/MNIST/' \
  --output '/workspace/output/=>user-tklee/experiments/demo-mnist/' \
  --gpu 1 --gpu-model h100 --memory 8Gi
```

기대:
- Build Job (Kaniko) 1-2분 (cache 미스면 5-10분)
- Run Job:
  - Init (mc cp): 수 초
  - Main (training): 1-2분
  - inline upload (결과): 수 초
- 결과: `lab/user-tklee/experiments/demo-mnist/<run-id>/mnist_cnn.pt` + `meta.txt`

## 확인

```bash
# 작업 진행 (Build → Run)
ssh dclserver75 "kubectl logs -n experiments -l aimimo.io/experiment-id=<id> -f --max-log-requests 5 --all-containers"

# 결과 download
mc ls lab/user-tklee/experiments/demo-mnist/
mc cp lab/user-tklee/experiments/demo-mnist/<run-id>/mnist_cnn.pt /tmp/
```
