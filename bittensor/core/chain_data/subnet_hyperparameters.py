import re
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from bittensor.core.chain_data.info_base import InfoBase
from bittensor.utils import Self
from bittensor.utils.balance import fixed_to_float

_FIXED_POINT_TAG = re.compile(r"^[UI]\d+F(\d+)$")

# Chain field name -> dataclass attribute (v2/v1 use ``max_weights_limit``).
_FIELD_ALIASES = {"max_weights_limit": "max_weight_limit"}


@dataclass
class SubnetHyperparameters(InfoBase):
    """
    Hyperparameters for a subnet.

    Known fields are explicit typed attributes for IDE support. Values returned
    by the chain under other names are stored in ``hyperparameters`` and are
    accessible via attribute, item, or mapping-style access.
    """

    rho: Optional[int] = None
    kappa: Optional[int] = None
    immunity_period: Optional[int] = None
    min_allowed_weights: Optional[int] = None
    max_weight_limit: Optional[float] = None
    tempo: Optional[int] = None
    min_difficulty: Optional[int] = None
    max_difficulty: Optional[int] = None
    weights_version: Optional[int] = None
    weights_rate_limit: Optional[int] = None
    adjustment_interval: Optional[int] = None
    activity_cutoff: Optional[int] = None
    registration_allowed: Optional[bool] = None
    target_regs_per_interval: Optional[int] = None
    min_burn: Optional[int] = None
    max_burn: Optional[int] = None
    bonds_moving_avg: Optional[int] = None
    max_regs_per_block: Optional[int] = None
    serving_rate_limit: Optional[int] = None
    max_validators: Optional[int] = None
    adjustment_alpha: Optional[int] = None
    difficulty: Optional[int] = None
    commit_reveal_period: Optional[int] = None
    commit_reveal_weights_enabled: Optional[bool] = None
    alpha_high: Optional[int] = None
    alpha_low: Optional[int] = None
    liquid_alpha_enabled: Optional[bool] = None
    alpha_sigmoid_steepness: Optional[float] = None
    yuma_version: Optional[int] = None
    subnet_is_active: Optional[bool] = None
    transfers_enabled: Optional[bool] = None
    bonds_reset_enabled: Optional[bool] = None
    user_liquidity_enabled: Optional[bool] = None
    activity_cutoff_factor: Optional[int] = None
    hyperparameters: dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def _typed_field_names() -> frozenset[str]:
        return frozenset(
            f.name for f in fields(SubnetHyperparameters) if f.name != "hyperparameters"
        )

    @staticmethod
    def _decode_value(value: Any) -> Any:
        """Decode a single ``{<type_tag>: <payload>}`` hyperparameter value."""
        if isinstance(value, dict) and set(value.keys()) == {"bits"}:
            # V2 struct encodes fixed-point fields as {"bits": N} without a type tag. All V2 fixed-point fields are
            # I32F32. V3 entries carry the enum variant tag and are handled below.
            return fixed_to_float(value["bits"], frac_bits=32)
        if not isinstance(value, dict) or len(value) != 1:
            return value
        ((type_tag, payload),) = value.items()
        if type_tag == "Bool":
            return bool(payload)
        if match := _FIXED_POINT_TAG.match(type_tag):
            if isinstance(payload, dict) and "bits" in payload:
                payload = payload["bits"]
            return fixed_to_float(payload, frac_bits=int(match.group(1)))
        try:
            return int(payload)
        except (TypeError, ValueError):
            return payload

    @classmethod
    def _fix_decoded(cls, decoded: list | dict | Self) -> Self:
        if isinstance(decoded, SubnetHyperparameters):
            return decoded

        if isinstance(decoded, dict):
            entries = decoded.items()
        else:
            entries = ((record["name"], record["value"]) for record in decoded)

        flat: dict[str, Any] = {}
        for name, value in entries:
            if isinstance(name, (bytes, bytearray)):
                name = name.decode("utf-8", errors="replace")
            if not isinstance(name, str):
                continue
            flat[name] = cls._decode_value(value)

        for chain_name, attr_name in _FIELD_ALIASES.items():
            if chain_name in flat and attr_name not in flat:
                flat[attr_name] = flat[chain_name]

        typed_names = cls._typed_field_names()
        typed_kwargs = {name: flat[name] for name in typed_names if name in flat}
        spillover = {
            name: value for name, value in flat.items() if name not in typed_names
        }
        return cls(hyperparameters=spillover, **typed_kwargs)

    @classmethod
    def from_any(cls, data: Any) -> Self:
        return cls._fix_decoded(data)

    @classmethod
    def from_dict(cls, decoded: dict) -> Self:
        return cls.from_any(decoded)

    def __getattr__(self, item: str) -> Any:
        try:
            return self.__dict__["hyperparameters"][item]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__!r} object has no hyperparameter {item!r}"
            )

    def __getitem__(self, item: str) -> Any:
        if item in self._typed_field_names():
            return getattr(self, item)
        return self.hyperparameters[item]

    def __iter__(self):
        return self.keys()

    def __contains__(self, item: str) -> bool:
        if item in self._typed_field_names():
            return getattr(self, item) is not None
        return item in self.hyperparameters

    def get(self, item: str, default: Any = None) -> Any:
        if item in self._typed_field_names():
            value = getattr(self, item)
            return default if value is None else value
        return self.hyperparameters.get(item, default)

    def items(self):
        for name in sorted(self._typed_field_names()):
            value = getattr(self, name)
            if value is not None:
                yield name, value
        yield from self.hyperparameters.items()

    def keys(self):
        return (name for name, _ in self.items())

    def values(self):
        return (value for _, value in self.items())
