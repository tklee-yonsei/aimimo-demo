"""
Phase 4 demo — 딥러닝 없는 간단한 소수(prime) 계산.

argument:
  limit               : 소수를 찾을 상한값 (positional)
                        한도 초과/이하 시 clamp:
                          limit > 1000000  → 1000000 으로 고정
                          limit < 2        → 2 로 고정

환경변수:
  AIMIMO_OUTPUT_DIR   : 결과 저장 dir (default: /workspace/output)
  AIMIMO_USER         : 실행자 (logging 용, optional)
  AIMIMO_EXPERIMENT_ID: 실험 ID (logging 용, optional)
"""

import argparse
import os
import time


def clamp_limit(limit: int) -> int:
    """상한값 한도 처리.

    1000000 을 넘어가면 1000000 으로 고정, 2 미만이면 2 로 고정.
    """
    if limit > 1000000:
        return 1000000
    if limit < 2:
        return 2
    return limit


def sieve_primes(limit: int) -> list[int]:
    """에라토스테네스의 체로 limit 이하의 소수 목록 반환."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    p = 2
    while p * p <= limit:
        if is_prime[p]:
            for multiple in range(p * p, limit + 1, p):
                is_prime[multiple] = False
        p += 1
    return [i for i, prime in enumerate(is_prime) if prime]


def main() -> None:
    parser = argparse.ArgumentParser(description="소수(prime) 계산 demo")
    parser.add_argument("limit", type=int, help="소수를 찾을 상한값")
    args = parser.parse_args()

    output_dir = os.environ.get("AIMIMO_OUTPUT_DIR", "/workspace/output")
    user = os.environ.get("AIMIMO_USER", "unknown")
    exp_id = os.environ.get("AIMIMO_EXPERIMENT_ID", "local")

    limit = clamp_limit(args.limit)

    print("=" * 60)
    print(f"  aimimo demo — Prime (에라토스테네스의 체)")
    print(f"  user      : {user}")
    print(f"  experiment: {exp_id}")
    print(f"  output dir: {output_dir}")
    print(f"  limit (입력) : {args.limit}")
    print(f"  limit (clamp): {limit}")
    print("=" * 60)

    start = time.time()
    primes = sieve_primes(limit)
    elapsed = time.time() - start
    print(f"  찾은 소수 개수: {len(primes)}")
    print(f"  가장 큰 소수  : {primes[-1]}")
    print(f"  소요 시간     : {elapsed:.3f}s")
    print()

    # 결과 저장 (Sidecar / inline 이 MinIO 로 upload)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "primes.txt")
    with open(out_path, "w") as f:
        for i, v in enumerate(primes):
            f.write(f"{i}\t{v}\n")
    print(f"  Saved → {out_path}")

    # 메타데이터 — 결과 reproducibility
    meta_path = os.path.join(output_dir, "meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"user={user}\n")
        f.write(f"experiment_id={exp_id}\n")
        f.write(f"limit_input={args.limit}\n")
        f.write(f"limit_clamped={limit}\n")
        f.write(f"prime_count={len(primes)}\n")
        f.write(f"largest_prime={primes[-1]}\n")
        f.write(f"elapsed_seconds={elapsed:.3f}\n")
    print(f"  Saved → {meta_path}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
