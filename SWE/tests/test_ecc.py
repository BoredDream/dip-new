# -*- coding: utf-8 -*-
"""纠错码单元测试。"""
import numpy as np
import pytest

from swe.ecc.codec import RepetitionECC, HammingECC, ReedSolomonECC, make_ecc


def _rand_bits(n, seed=0):
    return np.random.RandomState(seed).binomial(1, 0.5, n).astype(np.int64)


def test_repetition_roundtrip():
    ecc = RepetitionECC(5)
    data = _rand_bits(40)
    assert np.array_equal(ecc.decode_bits(ecc.encode_bits(data), len(data)), data)


def test_repetition_corrects_minority():
    ecc = RepetitionECC(5)
    data = _rand_bits(40, 1)
    coded = ecc.encode_bits(data).reshape(-1, 5)
    coded[:, 0] ^= 1  # 每组翻 1 位(少数)
    dec = ecc.decode_bits(coded.reshape(-1), len(data))
    assert np.array_equal(dec, data)


def test_hamming_corrects_one_per_block():
    ecc = HammingECC()
    data = _rand_bits(64, 2)
    coded = ecc.encode_bits(data).reshape(-1, 7)
    rng = np.random.RandomState(3)
    for r in range(coded.shape[0]):
        coded[r, rng.randint(7)] ^= 1
    dec = ecc.decode_bits(coded.reshape(-1), len(data))
    assert np.array_equal(dec, data)


def test_reed_solomon_no_error():
    ecc = ReedSolomonECC(nsym=10)
    data = _rand_bits(96, 4)
    assert np.array_equal(ecc.decode_bits(ecc.encode_bits(data), len(data)), data)


@pytest.mark.parametrize("nbyte", [1, 3, 5])
def test_reed_solomon_corrects_up_to_t(nbyte):
    ecc = ReedSolomonECC(nsym=10)  # 纠 5 字节错
    data = _rand_bits(96, 5)
    coded = ecc.encode_bits(data).reshape(-1, 8)
    rng = np.random.RandomState(10 + nbyte)
    idx = rng.choice(coded.shape[0], nbyte, replace=False)
    for i in idx:
        coded[i] = np.unpackbits(np.uint8([rng.randint(1, 256)]))
    dec = ecc.decode_bits(coded.reshape(-1), len(data))
    assert np.array_equal(dec, data)


def test_reed_solomon_overhead():
    ecc = ReedSolomonECC(nsym=10)
    data = _rand_bits(96)
    coded = ecc.encode_bits(data)
    assert len(coded) == (12 + 10) * 8  # 12 数据字节 + 10 校验字节


def test_make_ecc_factory():
    assert isinstance(make_ecc("rs"), ReedSolomonECC)
    assert isinstance(make_ecc("hamming"), HammingECC)
    assert isinstance(make_ecc("rep", repeat=3), RepetitionECC)
