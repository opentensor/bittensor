import factory

from bittensor.core.chain_data import AxonInfo, NeuronInfoLite, PrometheusInfo
from bittensor.utils.balance import Balance


class BalanceFactory(factory.Factory):
    class Meta:
        model = Balance

    balance = factory.Faker("pyfloat", left_digits=3, right_digits=6, positive=True)


class PrometheusInfoFactory(factory.Factory):
    class Meta:
        model = PrometheusInfo

    block = factory.Faker("random_int", min=0, max=100)
    version = factory.Faker("random_int", min=0, max=100)
    ip = factory.Faker("ipv4")
    port = factory.Faker("random_int", min=0, max=100)
    ip_type = factory.Faker("random_int", min=0, max=100)


class AxonInfoFactory(factory.Factory):
    class Meta:
        model = AxonInfo

    version = factory.Faker("random_int", min=0, max=100)
    ip = factory.Faker("ipv4")
    port = factory.Faker("random_int", min=0, max=100)
    ip_type = factory.Faker("random_int", min=0, max=100)
    hotkey = factory.Faker("uuid4")
    coldkey = factory.Faker("uuid4")


class NeuronInfoLiteFactory(factory.Factory):
    class Meta:
        model = NeuronInfoLite

    hotkey = factory.Faker("uuid4")
    coldkey = factory.Faker("uuid4")
    uid = factory.Sequence(lambda n: n)
    netuid = factory.Sequence(lambda n: n)
    active = factory.Faker("random_int", min=0, max=1)
    stake = factory.SubFactory(BalanceFactory)
    stake_dict = factory.Dict({"balance": 10})
    total_stake = factory.SubFactory(BalanceFactory)
    rank = factory.Faker("pyfloat", left_digits=3, right_digits=6, positive=True)
    emission = factory.Faker("pyfloat", left_digits=3, right_digits=6, positive=True)
    incentive = factory.Faker("pyfloat", left_digits=3, right_digits=6, positive=True)
    consensus = factory.Faker("pyfloat", left_digits=3, right_digits=6, positive=True)
    trust = factory.Faker("pyfloat", left_digits=3, right_digits=6, positive=True)
    validator_trust = factory.Faker(
        "pyfloat", left_digits=3, right_digits=6, positive=True
    )
    dividends = factory.Faker("pyfloat", left_digits=3, right_digits=6, positive=True)
    last_update = factory.Faker("unix_time")
    validator_permit = factory.Faker("boolean")
    prometheus_info = factory.SubFactory(PrometheusInfoFactory)
    axon_info = factory.SubFactory(AxonInfoFactory)
    pruning_score = factory.Faker("random_int", min=0, max=100)
    is_null = factory.Faker("boolean")
