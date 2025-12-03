"""Unit tests for bittensor.core.chain_data.neuron_info_lite module."""
import pytest
from unittest.mock import patch, MagicMock
from bittensor.core.chain_data.neuron_info_lite import NeuronInfoLite
from bittensor.core.chain_data.axon_info import AxonInfo
from bittensor.core.chain_data.prometheus_info import PrometheusInfo
from bittensor.utils.balance import Balance
MOCK_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
MOCK_COLDKEY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
NULL_KEY = "000000000000000000000000000000000000000000000000"
class TestNeuronInfoLiteCreation:
    """Tests for NeuronInfoLite dataclass creation."""
    def test_neuron_info_lite_basic_creation(self):
        """Test creating a NeuronInfoLite instance with valid data."""
        neuron = NeuronInfoLite(
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
            uid=1,
            netuid=1,
            active=1,
            stake=Balance.from_tao(100.0),
            stake_dict={MOCK_COLDKEY: Balance.from_tao(100.0)},
            total_stake=Balance.from_tao(100.0),
            rank=0.5,
            emission=0.001,
            incentive=0.3,
            consensus=0.8,
            trust=0.9,
            validator_trust=0.85,
            dividends=0.1,
            last_update=1000,
            validator_permit=True,
            prometheus_info=None,
            axon_info=None,
            pruning_score=100,
        )
        assert neuron.hotkey == MOCK_HOTKEY
        assert neuron.coldkey == MOCK_COLDKEY
        assert neuron.uid == 1
        assert neuron.netuid == 1
        assert neuron.stake.tao == 100.0
        assert neuron.validator_permit is True
        assert neuron.is_null is False
    def test_neuron_info_lite_with_axon_info(self):
        """Test NeuronInfoLite with AxonInfo attached."""
        axon = AxonInfo(
            version=1,
            ip="192.168.1.1",
            port=8080,
            ip_type=4,
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
        )
        neuron = NeuronInfoLite(
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
            uid=1,
            netuid=1,
            active=1,
            stake=Balance.from_tao(50.0),
            stake_dict={},
            total_stake=Balance.from_tao(50.0),
            rank=0.3,
            emission=0.0005,
            incentive=0.2,
            consensus=0.7,
            trust=0.8,
            validator_trust=0.75,
            dividends=0.05,
            last_update=500,
            validator_permit=False,
            prometheus_info=None,
            axon_info=axon,
            pruning_score=50,
        )
        assert neuron.axon_info is not None
        assert neuron.axon_info.ip == "192.168.1.1"
    def test_neuron_info_lite_with_prometheus_info(self):
        """Test NeuronInfoLite with PrometheusInfo attached."""
        prometheus = PrometheusInfo(
            block=1000,
            version=1,
            ip="10.0.0.1",
            port=9090,
            ip_type=4,
        )
        neuron = NeuronInfoLite(
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
            uid=2,
            netuid=1,
            active=1,
            stake=Balance.from_tao(200.0),
            stake_dict={},
            total_stake=Balance.from_tao(200.0),
            rank=0.7,
            emission=0.002,
            incentive=0.4,
            consensus=0.85,
            trust=0.95,
            validator_trust=0.9,
            dividends=0.15,
            last_update=2000,
            validator_permit=True,
            prometheus_info=prometheus,
            axon_info=None,
            pruning_score=200,
        )
        assert neuron.prometheus_info is not None
        assert neuron.prometheus_info.port == 9090
    def test_neuron_info_lite_zero_values(self):
        """Test NeuronInfoLite with zero values."""
        neuron = NeuronInfoLite(
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
            uid=0,
            netuid=0,
            active=0,
            stake=Balance.from_rao(0),
            stake_dict={},
            total_stake=Balance.from_rao(0),
            rank=0.0,
            emission=0.0,
            incentive=0.0,
            consensus=0.0,
            trust=0.0,
            validator_trust=0.0,
            dividends=0.0,
            last_update=0,
            validator_permit=False,
            prometheus_info=None,
            axon_info=None,
            pruning_score=0,
        )
        assert neuron.uid == 0
        assert neuron.rank == 0.0
        assert neuron.stake.rao == 0
class TestNeuronInfoLiteNullNeuron:
    """Tests for NeuronInfoLite.get_null_neuron method."""
    def test_get_null_neuron(self):
        """Test get_null_neuron returns a valid null neuron."""
        null_neuron = NeuronInfoLite.get_null_neuron()
        assert null_neuron.is_null is True
        assert null_neuron.uid == 0
        assert null_neuron.netuid == 0
        assert null_neuron.hotkey == NULL_KEY
        assert null_neuron.coldkey == NULL_KEY
        assert null_neuron.stake.rao == 0
    def test_null_neuron_all_attributes(self):
        """Test all attributes of null neuron are properly set."""
        null_neuron = NeuronInfoLite.get_null_neuron()
        assert null_neuron.active == 0
        assert null_neuron.rank == 0
        assert null_neuron.emission == 0
        assert null_neuron.incentive == 0
        assert null_neuron.consensus == 0
        assert null_neuron.trust == 0
        assert null_neuron.validator_trust == 0
        assert null_neuron.dividends == 0
        assert null_neuron.last_update == 0
        assert null_neuron.validator_permit is False
        assert null_neuron.pruning_score == 0
        assert null_neuron.prometheus_info is None
        assert null_neuron.axon_info is None
    def test_null_neuron_consistency(self):
        """Test that multiple calls return consistent null neurons."""
        null1 = NeuronInfoLite.get_null_neuron()
        null2 = NeuronInfoLite.get_null_neuron()
        assert null1.is_null == null2.is_null
        assert null1.uid == null2.uid
        assert null1.hotkey == null2.hotkey
        assert null1.coldkey == null2.coldkey
class TestNeuronInfoLiteFromDict:
    """Tests for NeuronInfoLite._from_dict method."""
    @patch("bittensor.core.chain_data.neuron_info_lite.decode_account_id")
    @patch("bittensor.core.chain_data.neuron_info_lite.process_stake_data")
    @patch("bittensor.core.chain_data.neuron_info_lite.AxonInfo.from_dict")
    @patch("bittensor.core.chain_data.neuron_info_lite.PrometheusInfo.from_dict")
    def test_from_dict_basic(
        self, mock_prometheus, mock_axon, mock_stake, mock_decode
    ):
        """Test _from_dict with basic valid data."""
        mock_decode.side_effect = lambda x: MOCK_HOTKEY if "hotkey" in str(x) else MOCK_COLDKEY
        mock_stake.return_value = {MOCK_COLDKEY: Balance.from_tao(100.0)}
        mock_axon.return_value = MagicMock()
        mock_prometheus.return_value = MagicMock()
        decoded = {
            "hotkey": "hotkey_bytes",
            "coldkey": "coldkey_bytes",
            "uid": 1,
            "netuid": 1,
            "active": 1,
            "stake": [],
            "rank": 32768,
            "emission": 1000000000,
            "incentive": 16384,
            "consensus": 49152,
            "trust": 57344,
            "validator_trust": 53248,
            "dividends": 8192,
            "last_update": 1000,
            "validator_permit": True,
            "pruning_score": 100,
            "axon_info": {},
            "prometheus_info": {},
        }
        result = NeuronInfoLite._from_dict(decoded)
        assert result is not None
        assert result.uid == 1
        assert result.netuid == 1
        assert result.active == 1
        assert result.validator_permit is True
    @patch("bittensor.core.chain_data.neuron_info_lite.decode_account_id")
    @patch("bittensor.core.chain_data.neuron_info_lite.process_stake_data")
    @patch("bittensor.core.chain_data.neuron_info_lite.AxonInfo.from_dict")
    @patch("bittensor.core.chain_data.neuron_info_lite.PrometheusInfo.from_dict")
    def test_from_dict_zero_stake(
        self, mock_prometheus, mock_axon, mock_stake, mock_decode
    ):
        """Test _from_dict with zero stake."""
        mock_decode.side_effect = lambda x: MOCK_HOTKEY
        mock_stake.return_value = {}
        mock_axon.return_value = None
        mock_prometheus.return_value = None
        decoded = {
            "hotkey": "hotkey_bytes",
            "coldkey": "coldkey_bytes",
            "uid": 0,
            "netuid": 0,
            "active": 0,
            "stake": [],
            "rank": 0,
            "emission": 0,
            "incentive": 0,
            "consensus": 0,
            "trust": 0,
            "validator_trust": 0,
            "dividends": 0,
            "last_update": 0,
            "validator_permit": False,
            "pruning_score": 0,
            "axon_info": {},
            "prometheus_info": {},
        }
        result = NeuronInfoLite._from_dict(decoded)
        assert result is not None
        assert result.stake.rao == 0
        assert result.stake_dict == {}
class TestNeuronInfoLiteStake:
    """Tests for stake-related functionality in NeuronInfoLite."""
    def test_stake_dict_single_coldkey(self):
        """Test NeuronInfoLite with single coldkey in stake_dict."""
        stake_dict = {MOCK_COLDKEY: Balance.from_tao(500.0)}
        neuron = NeuronInfoLite(
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
            uid=1,
            netuid=1,
            active=1,
            stake=Balance.from_tao(500.0),
            stake_dict=stake_dict,
            total_stake=Balance.from_tao(500.0),
            rank=0.5,
            emission=0.001,
            incentive=0.3,
            consensus=0.8,
            trust=0.9,
            validator_trust=0.85,
            dividends=0.1,
            last_update=1000,
            validator_permit=True,
            prometheus_info=None,
            axon_info=None,
            pruning_score=100,
        )
        assert len(neuron.stake_dict) == 1
        assert neuron.stake_dict[MOCK_COLDKEY].tao == 500.0
    def test_stake_dict_multiple_coldkeys(self):
        """Test NeuronInfoLite with multiple coldkeys in stake_dict."""
        other_coldkey = "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy"
        stake_dict = {
            MOCK_COLDKEY: Balance.from_tao(300.0),
            other_coldkey: Balance.from_tao(200.0),
        }
        total = Balance.from_tao(500.0)
        neuron = NeuronInfoLite(
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
            uid=1,
            netuid=1,
            active=1,
            stake=total,
            stake_dict=stake_dict,
            total_stake=total,
            rank=0.5,
            emission=0.001,
            incentive=0.3,
            consensus=0.8,
            trust=0.9,
            validator_trust=0.85,
            dividends=0.1,
            last_update=1000,
            validator_permit=True,
            prometheus_info=None,
            axon_info=None,
            pruning_score=100,
        )
        assert len(neuron.stake_dict) == 2
        total_from_dict = sum(neuron.stake_dict.values())
        assert total_from_dict.tao == 500.0
    def test_total_stake_equals_stake(self):
        """Test that total_stake equals stake when properly set."""
        stake_amount = Balance.from_tao(250.0)
        neuron = NeuronInfoLite(
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
            uid=1,
            netuid=1,
            active=1,
            stake=stake_amount,
            stake_dict={MOCK_COLDKEY: stake_amount},
            total_stake=stake_amount,
            rank=0.5,
            emission=0.001,
            incentive=0.3,
            consensus=0.8,
            trust=0.9,
            validator_trust=0.85,
            dividends=0.1,
            last_update=1000,
            validator_permit=True,
            prometheus_info=None,
            axon_info=None,
            pruning_score=100,
        )
        assert neuron.stake == neuron.total_stake
class TestNeuronInfoLiteMetrics:
    """Tests for metric values in NeuronInfoLite."""
    def test_normalized_metrics_range(self):
        """Test that normalized metrics are within valid range."""
        neuron = NeuronInfoLite(
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
            uid=1,
            netuid=1,
            active=1,
            stake=Balance.from_tao(100.0),
            stake_dict={},
            total_stake=Balance.from_tao(100.0),
            rank=0.5,
            emission=0.001,
            incentive=0.3,
            consensus=0.8,
            trust=0.9,
            validator_trust=0.85,
            dividends=0.1,
            last_update=1000,
            validator_permit=True,
            prometheus_info=None,
            axon_info=None,
            pruning_score=100,
        )
        assert 0.0 <= neuron.rank <= 1.0
        assert 0.0 <= neuron.incentive <= 1.0
        assert 0.0 <= neuron.consensus <= 1.0
        assert 0.0 <= neuron.trust <= 1.0
        assert 0.0 <= neuron.validator_trust <= 1.0
        assert 0.0 <= neuron.dividends <= 1.0
    def test_emission_value(self):
        """Test emission value handling."""
        neuron = NeuronInfoLite(
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
            uid=1,
            netuid=1,
            active=1,
            stake=Balance.from_tao(100.0),
            stake_dict={},
            total_stake=Balance.from_tao(100.0),
            rank=0.5,
            emission=0.00123456,
            incentive=0.3,
            consensus=0.8,
            trust=0.9,
            validator_trust=0.85,
            dividends=0.1,
            last_update=1000,
            validator_permit=True,
            prometheus_info=None,
            axon_info=None,
            pruning_score=100,
        )
        assert neuron.emission == 0.00123456
    def test_last_update_timestamp(self):
        """Test last_update timestamp handling."""
        timestamps = [0, 1000, 100000, 1000000000]
        for ts in timestamps:
            neuron = NeuronInfoLite(
                hotkey=MOCK_HOTKEY,
                coldkey=MOCK_COLDKEY,
                uid=1,
                netuid=1,
                active=1,
                stake=Balance.from_tao(100.0),
                stake_dict={},
                total_stake=Balance.from_tao(100.0),
                rank=0.5,
                emission=0.001,
                incentive=0.3,
                consensus=0.8,
                trust=0.9,
                validator_trust=0.85,
                dividends=0.1,
                last_update=ts,
                validator_permit=True,
                prometheus_info=None,
                axon_info=None,
                pruning_score=100,
            )
            assert neuron.last_update == ts
class TestNeuronInfoLiteComparison:
    """Tests for comparing NeuronInfoLite instances."""
    def test_different_uids(self):
        """Test neurons with different UIDs."""
        neuron1 = NeuronInfoLite(
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
            uid=1,
            netuid=1,
            active=1,
            stake=Balance.from_tao(100.0),
            stake_dict={},
            total_stake=Balance.from_tao(100.0),
            rank=0.5,
            emission=0.001,
            incentive=0.3,
            consensus=0.8,
            trust=0.9,
            validator_trust=0.85,
            dividends=0.1,
            last_update=1000,
            validator_permit=True,
            prometheus_info=None,
            axon_info=None,
            pruning_score=100,
        )
        neuron2 = NeuronInfoLite(
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
            uid=2,
            netuid=1,
            active=1,
            stake=Balance.from_tao(100.0),
            stake_dict={},
            total_stake=Balance.from_tao(100.0),
            rank=0.5,
            emission=0.001,
            incentive=0.3,
            consensus=0.8,
            trust=0.9,
            validator_trust=0.85,
            dividends=0.1,
            last_update=1000,
            validator_permit=True,
            prometheus_info=None,
            axon_info=None,
            pruning_score=100,
        )
        assert neuron1.uid != neuron2.uid
    def test_different_netuids(self):
        """Test neurons on different subnets."""
        neuron1 = NeuronInfoLite(
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
            uid=1,
            netuid=1,
            active=1,
            stake=Balance.from_tao(100.0),
            stake_dict={},
            total_stake=Balance.from_tao(100.0),
            rank=0.5,
            emission=0.001,
            incentive=0.3,
            consensus=0.8,
            trust=0.9,
            validator_trust=0.85,
            dividends=0.1,
            last_update=1000,
            validator_permit=True,
            prometheus_info=None,
            axon_info=None,
            pruning_score=100,
        )
        neuron2 = NeuronInfoLite(
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
            uid=1,
            netuid=2,
            active=1,
            stake=Balance.from_tao(100.0),
            stake_dict={},
            total_stake=Balance.from_tao(100.0),
            rank=0.5,
            emission=0.001,
            incentive=0.3,
            consensus=0.8,
            trust=0.9,
            validator_trust=0.85,
            dividends=0.1,
            last_update=1000,
            validator_permit=True,
            prometheus_info=None,
            axon_info=None,
            pruning_score=100,
        )
        assert neuron1.netuid != neuron2.netuid
