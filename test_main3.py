"""
main3.py 유닛 테스트 — 딥러닝 없이 stdlib unittest 로 실행.

    python3 -m unittest test_main3 -v
"""

import unittest

from main3 import clamp_limit, sieve_primes


class ClampLimitTest(unittest.TestCase):
    def test_within_range_unchanged(self) -> None:
        self.assertEqual(clamp_limit(50), 50)
        self.assertEqual(clamp_limit(2), 2)
        self.assertEqual(clamp_limit(1000000), 1000000)

    def test_above_upper_bound_clamped(self) -> None:
        self.assertEqual(clamp_limit(1000001), 1000000)
        self.assertEqual(clamp_limit(9999999), 1000000)

    def test_below_lower_bound_clamped(self) -> None:
        self.assertEqual(clamp_limit(1), 2)
        self.assertEqual(clamp_limit(0), 2)
        self.assertEqual(clamp_limit(-10), 2)


class SievePrimesTest(unittest.TestCase):
    def test_small_limit(self) -> None:
        self.assertEqual(sieve_primes(10), [2, 3, 5, 7])
        self.assertEqual(sieve_primes(2), [2])

    def test_limit_50(self) -> None:
        primes = sieve_primes(50)
        self.assertEqual(len(primes), 15)
        self.assertEqual(primes[-1], 47)

    def test_inclusive_upper_bound(self) -> None:
        # 상한값 자체가 소수면 포함되어야 함
        self.assertIn(13, sieve_primes(13))
        # 상한값이 합성수면 포함되지 않아야 함
        self.assertNotIn(14, sieve_primes(14))

    def test_all_returned_are_prime(self) -> None:
        primes = set(sieve_primes(100))
        for n in range(2, 101):
            is_prime = all(n % d for d in range(2, int(n ** 0.5) + 1))
            self.assertEqual(n in primes, is_prime, f"n={n} 판정 불일치")


if __name__ == "__main__":
    unittest.main()
