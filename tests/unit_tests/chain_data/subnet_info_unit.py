"""Unit tests for bittensor.core.chain_data.subnet_info module."""
import pytest
from unittest.mock import patch
from bittensor.core.chain_data.subnet_info import SubnetInfo
from bittensor.utils.balance import Balance


MOCK_OWNER = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


def create_subnet_info(**kwargs):
    """Helper function to create SubnetInfo with default values."""
    defaults = {
        "netuid": 1,
        "rho": 10,
        "kappa": 32767,
        "difficulty": 1000000,
        "immunity_period": 4096,
        "max_allowed_validators": 64,
        "min_allowed_weights": 1,
        "max_weight_limit": 0.5,
        "scaling_law_power": 50.0,
        "subnetwork_n": 256,
        "max_n": 4096,
        "blocks_since_epoch": 100,
        "tempo": 360,
        "modality": 0,
        "connection_requirements": {},
        "emission_value": 1.0,
        "burn": Balance.from_tao(1.0),
        "owner_ss58": MOCK_OWNER,
    }
    defaults.update(kwargs)
    return SubnetInfo(**defaults)


class TestSubnetInfoCreation:
    """Tests for SubnetInfo dataclass creation."""

    def test_subnet_info_basic_creation(self):
        """Test creating a SubnetInfo instance with valid data."""
        subnet = create_subnet_info()
        assert subnet.netuid == 1
        assert subnet.rho == 10
        assert subnet.kappa == 32767
        assert subnet.max_allowed_validators == 64
        assert subnet.tempo == 360
        assert subnet.owner_ss58 == MOCK_OWNER

    def test_subnet_info_zero_netuid(self):
        """Test SubnetInfo with netuid 0 (root network)."""
        subnet = create_subnet_info(netuid=0, emission_value=0.0)
        assert subnet.netuid == 0

    def test_subnet_info_high_netuid(self):
        """Test SubnetInfo with high netuid value."""
        subnet = create_subnet_info(netuid=255, max_n=8192)
        assert subnet.netuid == 255

    def test_subnet_info_with_connection_requirements(self):
        """Test SubnetInfo with connection_requirements set."""
        conn_req = {"1": 0.5, "2": 0.75}
        subnet = create_subnet_info(connection_requirements=conn_req)
        assert subnet.connection_requirements == conn_req


class TestSubnetInfoParameters:
    """Tests for SubnetInfo parameter values."""

    def test_rho_values(self):
        """Test SubnetInfo with various rho values."""
        for rho in [1, 10, 50, 100]:
            subnet = create_subnet_info(rho=rho)
            assert subnet.rho == rho

    def test_kappa_values(self):
        """Test SubnetInfo with various kappa values."""
        for kappa in [0, 16384, 32767, 65535]:
            subnet = create_subnet_info(kappa=kappa)
            assert subnet.kappa == kappa

    def test_tempo_values(self):
        """Test SubnetInfo with various tempo values."""
        for tempo in [99, 360, 720, 1440]:
            subnet = create_subnet_info(tempo=tempo)
            assert subnet.tempo == tempo

    def test_difficulty_values(self):
        """Test SubnetInfo with various difficulty values."""
        for difficulty in [1000, 1000000, 1000000000]:
            subnet = create_subnet_info(difficulty=difficulty)
            assert subnet.difficulty == difficulty


class TestSubnetInfoValidators:
    """Tests for validator-related parameters in SubnetInfo."""

    def test_max_allowed_validators(self):
        """Test SubnetInfo with various max_allowed_validators values."""
        for max_validators in [16, 32, 64, 128, 256]:
            subnet = create_subnet_info(max_allowed_validators=max_validators)
            assert subnet.max_allowed_validators == max_validators

    def test_min_allowed_weights(self):
        """Test SubnetInfo with various min_allowed_weights values."""
        for min_weights in [0, 1, 10, 100]:
            subnet = create_subnet_info(min_allowed_weights=min_weights)
            assert subnet.min_allowed_weights == min_weights

    def test_max_weight_limit(self):
        """Test SubnetInfo with various max_weight_limit values."""
        for max_weight in [0.1, 0.5, 0.75, 1.0]:
            subnet = create_subnet_info(max_weight_limit=max_weight)
            assert subnet.max_weight_limit == max_weight


class TestSubnetInfoNetwork:
    """Tests for network-related parameters in SubnetInfo."""

    def test_subnetwork_n_values(self):
        """Test SubnetInfo with various subnetwork_n values."""
        for n in [64, 128, 256, 512, 1024]:
            subnet = create_subnet_info(subnetwork_n=n)
            assert subnet.subnetwork_n == n

    def test_max_n_values(self):
        """Test SubnetInfo with various max_n values."""
        for max_n in [256, 1024, 4096, 16384]:
            subnet = create_subnet_info(max_n=max_n)
            assert subnet.max_n == max_n

    def test_modality_values(self):
        """Test SubnetInfo with various modality values."""
        for modality in [0, 1, 2]:
            subnet = create_subnet_info(modality=modality)
            assert subnet.modality == modality


class TestSubnetInfoEmission:
    """Tests for emission-related parameters in SubnetInfo."""

    def test_emission_value(self):
        """Test SubnetInfo with various emission values."""
        for emission in [0.0, 0.5, 1.0, 2.0]:
            subnet = create_subnet_info(emission_value=emission)
            assert subnet.emission_value == emission

    def test_burn_balance(self):
        """Test SubnetInfo with various burn values."""
        for burn_tao in [0.0, 0.5, 1.0, 10.0, 100.0]:
            subnet = create_subnet_info(burn=Balance.from_tao(burn_tao))
            assert subnet.burn.tao == burn_tao


class TestSubnetInfoFromDict:
    """Tests for SubnetInfo._from_dict method."""

    @patch("bittensor.core.chain_data.subnet_info.decode_account_id")
    def test_from_dict_basic(self, mock_decode):
        """Test _from_dict with basic valid data."""
        mock_decode.return_value = MOCK_OWNER
        decoded = {
            "netuid": 1,
            "rho": 10,
            "kappa": 32767,
            "difficulty": 1000000,
            "immunity_period": 4096,
            "max_allowed_validators": 64,
            "min_allowed_weights": 1,
            "max_weights_limit": 32768,
            "scaling_law_power": 50,
            "subnetwork_n": 256,
            "max_allowed_uids": 4096,
            "blocks_since_last_step": 100,
            "tempo": 360,
            "network_modality": 0,
            "network_connect": [],
            "emission_value": 1.0,
            "burn": 1000000000,
            "owner": "owner_bytes",
        }
        result = SubnetInfo._from_dict(decoded)
        assert result.netuid == 1
        assert result.rho == 10
        assert result.tempo == 360
        assert result.owner_ss58 == MOCK_OWNER

    @patch("bittensor.core.chain_data.subnet_info.decode_account_id")
    def test_from_dict_with_network_connect(self, mock_decode):
        """Test _from_dict with network_connect data."""
        mock_decode.return_value = MOCK_OWNER
        decoded = {
            "netuid": 1,
            "rho": 10,
            "kappa": 32767,
            "difficulty": 1000000,
            "immunity_period": 4096,
            "max_allowed_validators": 64,
            "min_allowed_weights": 1,
            "max_weights_limit": 32768,
            "scaling_law_power": 50,
            "subnetwork_n": 256,
            "max_allowed_uids": 4096,
            "blocks_since_last_step": 100,
            "tempo": 360,
            "network_modality": 0,
            "network_connect": [(2, 32768), (3, 16384)],
            "emission_value": 1.0,
            "burn": 1000000000,
            "owner": "owner_bytes",
        }
        result = SubnetInfo._from_dict(decoded)
        assert "2" in result.connection_requirements
        assert "3" in result.connection_requirements


class TestSubnetInfoAttributes:
    """Tests for SubnetInfo attribute access."""

    def test_all_attributes_accessible(self):
        """Test that all attributes are accessible."""
        subnet = create_subnet_info()
        attrs = [
            "netuid",
            "rho",
            "kappa",
            "difficulty",
            "immunity_period",
            "max_allowed_validators",
            "min_allowed_weights",
            "max_weight_limit",
            "scaling_law_power",
            "subnetwork_n",
            "max_n",
            "blocks_since_epoch",
            "tempo",
            "modality",
            "connection_requirements",
            "emission_value",
            "burn",
            "owner_ss58",
        ]
        for attr in attrs:
            assert hasattr(subnet, attr)

    def test_owner_ss58_format(self):
        """Test that owner_ss58 is in correct format."""
        subnet = create_subnet_info()
        assert subnet.owner_ss58.startswith("5")
        assert len(subnet.owner_ss58) == 48

    def test_burn_is_balance(self):
        """Test that burn is a Balance object."""
        subnet = create_subnet_info()
        assert isinstance(subnet.burn, Balance)

    def test_connection_requirements_is_dict(self):
        """Test that connection_requirements is a dict."""
        subnet = create_subnet_info()
        assert isinstance(subnet.connection_requirements, dict)


class TestSubnetInfoComparison:
    """Tests for comparing SubnetInfo instances."""

    def test_different_netuids(self):
        """Test SubnetInfo instances with different netuids."""
        subnet1 = create_subnet_info(netuid=1)
        subnet2 = create_subnet_info(netuid=2)
        assert subnet1.netuid != subnet2.netuid

    def test_same_netuid_different_params(self):
        """Test SubnetInfo instances with same netuid but different params."""
        subnet1 = create_subnet_info(netuid=1, tempo=360)
        subnet2 = create_subnet_info(netuid=1, tempo=720)
        assert subnet1.netuid == subnet2.netuid
        assert subnet1.tempo != subnet2.tempo
