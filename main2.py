"""
Phase 4 demo — 딥러닝 없는 간단한 Fibonacci 계산.

argument:
  n                   : 계산할 피보나치 항 개수 (positional)
                        한도 초과/이하 시 clamp:
                          n > 100  → 200 으로 고정
                          n <= 0   → 1 로 고정

환경변수:
  AIMIMO_OUTPUT_DIR   : 결과 저장 dir (default: /workspace/output)
  AIMIMO_USER         : 실행자 (logging 용, optional)
  AIMIMO_EXPERIMENT_ID: 실험 ID (logging 용, optional)
"""

import argparse
import os


def clamp_n(n: int) -> int:
    """항 개수 한도 처리.

    100 을 넘어가면 200 으로 고정, 0 이하이면 1 로 고정.
    """
    if n > 100:
        return 200
    if n <= 0:
        return 1
    return n


def fibonacci(n: int) -> list[int]:
    """0, 1, 1, 2, ... 순서로 n 개의 피보나치 수열 반환."""
    seq: list[int] = []
    a, b = 0, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq


def main() -> None:
    parser = argparse.ArgumentParser(description="Fibonacci 계산 demo")
    parser.add_argument("n", type=int, help="계산할 피보나치 항 개수")
    args = parser.parse_args()

    output_dir = os.environ.get("AIMIMO_OUTPUT_DIR", "/workspace/output")
    user = os.environ.get("AIMIMO_USER", "unknown")
    exp_id = os.environ.get("AIMIMO_EXPERIMENT_ID", "local")

    n = clamp_n(args.n)

    print("=" * 60)
    print(f"  aimimo demo — Fibonacci")
    print(f"  user      : {user}")
    print(f"  experiment: {exp_id}")
    print(f"  output dir: {output_dir}")
    print(f"  n (입력)  : {args.n}")
    print(f"  n (clamp) : {n}")
    print("=" * 60)

    seq = fibonacci(n)
    print(f"  계산된 항 수: {len(seq)}")
    print(f"  마지막 값   : {seq[-1]}")
    print()

    # 결과 저장 (Sidecar / inline 이 MinIO 로 upload)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "fibonacci.txt")
    with open(out_path, "w") as f:
        for i, v in enumerate(seq):
            f.write(f"{i}\t{v}\n")
    print(f"  Saved → {out_path}")

    # 메타데이터 — 결과 reproducibility
    meta_path = os.path.join(output_dir, "meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"user={user}\n")
        f.write(f"experiment_id={exp_id}\n")
        f.write(f"n_input={args.n}\n")
        f.write(f"n_clamped={n}\n")
        f.write(f"last_value={seq[-1]}\n")
    print(f"  Saved → {meta_path}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
