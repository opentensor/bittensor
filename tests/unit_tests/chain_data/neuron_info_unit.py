"""Unit tests for bittensor.core.chain_data.neuron_info module."""
import pytest
from unittest.mock import patch, MagicMock
from bittensor.core.chain_data.neuron_info import NeuronInfo
from bittensor.core.chain_data.axon_info import AxonInfo
from bittensor.core.chain_data.prometheus_info import PrometheusInfo
from bittensor.utils.balance import Balance
MOCK_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
MOCK_COLDKEY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
NULL_KEY = "000000000000000000000000000000000000000000000000"
class TestNeuronInfoCreation:
    """Tests for NeuronInfo dataclass creation."""
    def test_neuron_info_basic_creation(self):
        """Test creating a NeuronInfo instance with valid data."""
        neuron = NeuronInfo(
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
            weights=[(0, 100), (1, 200)],
            bonds=[[0, 50], [1, 75]],
            pruning_score=100,
        )
        assert neuron.hotkey == MOCK_HOTKEY
        assert neuron.coldkey == MOCK_COLDKEY
        assert neuron.uid == 1
        assert neuron.netuid == 1
        assert neuron.active == 1
        assert neuron.stake.tao == 100.0
        assert neuron.rank == 0.5
        assert neuron.validator_permit is True
    def test_neuron_info_with_axon_info(self):
        """Test NeuronInfo with AxonInfo attached."""
        axon = AxonInfo(
            version=1,
            ip="192.168.1.1",
            port=8080,
            ip_type=4,
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
        )
        neuron = NeuronInfo(
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
            weights=[],
            bonds=[],
            pruning_score=50,
            axon_info=axon,
        )
        assert neuron.axon_info is not None
        assert neuron.axon_info.ip == "192.168.1.1"
        assert neuron.axon_info.port == 8080
    def test_neuron_info_with_prometheus_info(self):
        """Test NeuronInfo with PrometheusInfo attached."""
        prometheus = PrometheusInfo(
            block=1000,
            version=1,
            ip="10.0.0.1",
            port=9090,
            ip_type=4,
        )
        neuron = NeuronInfo(
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
            weights=[(0, 500)],
            bonds=[[0, 250]],
            pruning_score=200,
            prometheus_info=prometheus,
        )
        assert neuron.prometheus_info is not None
        assert neuron.prometheus_info.port == 9090
    def test_neuron_info_zero_values(self):
        """Test NeuronInfo with zero values."""
        neuron = NeuronInfo(
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
            weights=[],
            bonds=[],
            pruning_score=0,
        )
        assert neuron.uid == 0
        assert neuron.rank == 0.0
        assert neuron.stake.rao == 0
class TestNeuronInfoNullNeuron:
    """Tests for NeuronInfo.get_null_neuron method."""
    def test_get_null_neuron(self):
        """Test get_null_neuron returns a valid null neuron."""
        null_neuron = NeuronInfo.get_null_neuron()
        assert null_neuron.is_null is True
        assert null_neuron.uid == 0
        assert null_neuron.netuid == 0
        assert null_neuron.hotkey == NULL_KEY
        assert null_neuron.coldkey == NULL_KEY
        assert null_neuron.stake.rao == 0
        assert null_neuron.weights == []
        assert null_neuron.bonds == []
    def test_null_neuron_attributes(self):
        """Test all attributes of null neuron are properly set."""
        null_neuron = NeuronInfo.get_null_neuron()
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
    def test_null_neuron_is_singleton_like(self):
        """Test that multiple calls return equivalent null neurons."""
        null1 = NeuronInfo.get_null_neuron()
        null2 = NeuronInfo.get_null_neuron()
        assert null1.is_null == null2.is_null
        assert null1.uid == null2.uid
        assert null1.hotkey == null2.hotkey
class TestNeuronInfoFromWeightsBonds:
    """Tests for NeuronInfo.from_weights_bonds_and_neuron_lite method."""
    def test_from_weights_bonds_basic(self):
        """Test creating NeuronInfo from NeuronInfoLite with weights and bonds."""
        from bittensor.core.chain_data.neuron_info_lite import NeuronInfoLite
        neuron_lite = NeuronInfoLite(
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
            uid=5,
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
        weights_dict = {5: [(0, 100), (1, 200), (2, 150)]}
        bonds_dict = {5: [(0, 50), (1, 75)]}
        neuron = NeuronInfo.from_weights_bonds_and_neuron_lite(
            neuron_lite, weights_dict, bonds_dict
        )
        assert neuron.uid == 5
        assert neuron.weights == [(0, 100), (1, 200), (2, 150)]
        assert neuron.bonds == [(0, 50), (1, 75)]
        assert neuron.hotkey == MOCK_HOTKEY
    def test_from_weights_bonds_empty_dicts(self):
        """Test with empty weights and bonds dictionaries."""
        from bittensor.core.chain_data.neuron_info_lite import NeuronInfoLite
        neuron_lite = NeuronInfoLite(
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
            uid=10,
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
            axon_info=None,
            pruning_score=50,
        )
        neuron = NeuronInfo.from_weights_bonds_and_neuron_lite(
            neuron_lite, {}, {}
        )
        assert neuron.weights == []
        assert neuron.bonds == []
    def test_from_weights_bonds_uid_not_in_dict(self):
        """Test when neuron UID is not in weights/bonds dicts."""
        from bittensor.core.chain_data.neuron_info_lite import NeuronInfoLite
        neuron_lite = NeuronInfoLite(
            hotkey=MOCK_HOTKEY,
            coldkey=MOCK_COLDKEY,
            uid=99,
            netuid=1,
            active=1,
            stake=Balance.from_tao(25.0),
            stake_dict={},
            total_stake=Balance.from_tao(25.0),
            rank=0.1,
            emission=0.0001,
            incentive=0.1,
            consensus=0.5,
            trust=0.6,
            validator_trust=0.55,
            dividends=0.02,
            last_update=100,
            validator_permit=False,
            prometheus_info=None,
            axon_info=None,
            pruning_score=25,
        )
        weights_dict = {1: [(0, 100)], 2: [(0, 200)]}
        bonds_dict = {1: [(0, 50)]}
        neuron = NeuronInfo.from_weights_bonds_and_neuron_lite(
            neuron_lite, weights_dict, bonds_dict
        )
        assert neuron.weights == []
        assert neuron.bonds == []
class TestNeuronInfoFromDict:
    """Tests for NeuronInfo._from_dict method."""
    @patch("bittensor.core.chain_data.neuron_info.decode_account_id")
    @patch("bittensor.core.chain_data.neuron_info.process_stake_data")
    @patch("bittensor.core.chain_data.neuron_info.AxonInfo.from_dict")
    @patch("bittensor.core.chain_data.neuron_info.PrometheusInfo.from_dict")
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
            "weights": [(0, 100), (1, 200)],
            "bonds": [(0, 50), (1, 75)],
            "pruning_score": 100,
            "axon_info": {},
            "prometheus_info": {},
        }
        result = NeuronInfo._from_dict(decoded)
        assert result is not None
        assert result.uid == 1
        assert result.netuid == 1
        assert result.active == 1
        assert result.validator_permit is True
class TestNeuronInfoStake:
    """Tests for stake-related functionality in NeuronInfo."""
    def test_stake_dict_single_coldkey(self):
        """Test NeuronInfo with single coldkey in stake_dict."""
        stake_dict = {MOCK_COLDKEY: Balance.from_tao(500.0)}
        neuron = NeuronInfo(
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
            weights=[],
            bonds=[],
            pruning_score=100,
        )
        assert len(neuron.stake_dict) == 1
        assert neuron.stake_dict[MOCK_COLDKEY].tao == 500.0
    def test_stake_dict_multiple_coldkeys(self):
        """Test NeuronInfo with multiple coldkeys in stake_dict."""
        other_coldkey = "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy"
        stake_dict = {
            MOCK_COLDKEY: Balance.from_tao(300.0),
            other_coldkey: Balance.from_tao(200.0),
        }
        total = Balance.from_tao(500.0)
        neuron = NeuronInfo(
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
            weights=[],
            bonds=[],
            pruning_score=100,
        )
        assert len(neuron.stake_dict) == 2
        total_from_dict = sum(neuron.stake_dict.values())
        assert total_from_dict.tao == 500.0
class TestNeuronInfoWeightsBonds:
    """Tests for weights and bonds in NeuronInfo."""
    def test_weights_format(self):
        """Test weights are stored as list of tuples."""
        weights = [(0, 100), (1, 200), (2, 150), (3, 50)]
        neuron = NeuronInfo(
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
            weights=weights,
            bonds=[],
            pruning_score=100,
        )
        assert neuron.weights == weights
        assert len(neuron.weights) == 4
        assert all(isinstance(w, tuple) and len(w) == 2 for w in neuron.weights)
    def test_bonds_format(self):
        """Test bonds are stored as list of lists."""
        bonds = [[0, 50], [1, 75], [2, 100]]
        neuron = NeuronInfo(
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
            weights=[],
            bonds=bonds,
            pruning_score=100,
        )
        assert neuron.bonds == bonds
        assert len(neuron.bonds) == 3
