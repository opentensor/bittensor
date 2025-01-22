import pytest

from bittensor.utils.balance import fixed_to_float, FixedPoint

# Generated using the following gist: https://gist.github.com/camfairchild/8c6b6b9faa8aa1ae7ddc49ce177a27f2
examples: list[tuple[int, float]] = [
    (22773757908449605611411210240, 1234567890),
    (22773757910726980065558528000, 1234567890.1234567),
    (22773757910726980065558528000, 1234567890.1234567),
    (22773757910726980065558528000, 1234567890.1234567),
    (4611686018427387904, 0.25),
    (9223372036854775808, 0.5),
    (13835058055282163712, 0.75),
    (18446744073709551616, 1.0),
    (23058430092136939520, 1.25),
    (27670116110564327424, 1.5),
    (32281802128991715328, 1.75),
    (36893488147419103232, 2.0),
    (6148914691236516864, 0.3333333333333333),
    (2635249153387078656, 0.14285714285714285),
    (4611686018427387904, 0.25),
    (0, 0),
    (0, 0.0),
]


@pytest.mark.parametrize("bits, float_value", examples)
def test_fixed_to_float(bits: int, float_value: float):
    EPS = 1e-10
    fp = FixedPoint(bits=bits)
    assert abs(fixed_to_float(fp) - float_value) < EPS
