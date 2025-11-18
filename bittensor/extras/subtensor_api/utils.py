from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bittensor.extras import SubtensorApi


def add_legacy_methods(subtensor: "SubtensorApi"):
    """If SubtensorApi get `subtensor_fields=True` arguments, then all classic Subtensor fields added to root level."""
    # Attributes that should NOT be dynamically added (manually defined in SubtensorApi.__init__)
    EXCLUDED_ATTRIBUTES = {
        # Internal attributes
        "inner_subtensor",
        "initialize",
    }

    # Get all attributes from inner_subtensor
    for attr_name in dir(subtensor.inner_subtensor):
        # Skip private attributes, special methods, and excluded attributes
        if attr_name.startswith("_") or attr_name in EXCLUDED_ATTRIBUTES:
            continue

        # Check if attribute already exists in subtensor (this automatically excludes
        # all properties like block, chain, commitments, etc. and other defined attributes)
        if hasattr(subtensor, attr_name):
            continue

        # Get the attribute from inner_subtensor and add it
        try:
            attr_value = getattr(subtensor.inner_subtensor, attr_name)
            setattr(subtensor, attr_name, attr_value)
        except (AttributeError, TypeError):
            # Skip if attribute cannot be accessed or set
            continue
