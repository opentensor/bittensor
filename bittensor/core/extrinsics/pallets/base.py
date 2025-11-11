from dataclasses import dataclass

from typing import Awaitable, Callable, Union, TYPE_CHECKING

from scalecodec import GenericCall
from bittensor.utils import get_caller_name

if TYPE_CHECKING:
    from bittensor.core.async_subtensor import AsyncSubtensor
    from bittensor.core.subtensor import Subtensor


Call = Union[GenericCall, Awaitable[GenericCall]]


@dataclass
class CallBuilder:
    """Base class for creating GenericCall objects for all Subtensor pallet functions.

    This class implements an interface for creating GenericCall objects that can be used with any Subtensor pallet
    function. For async operations, pass an AsyncSubtensor instance and await the result.

    Attributes:
        subtensor: The Subtensor or AsyncSubtensor instance used for call composition.
        dynamic_function: If True, allows dynamic calls to functions not explicitly defined in the pallet class. When a
        method is called that doesn't exist in the class, it will be dynamically created as a call to the pallet
        function with the same name.
    """

    subtensor: Union["Subtensor", "AsyncSubtensor"]
    dynamic_function: bool = True

    def create_composed_call(
        self, call_module: str = None, call_function: str = None, **kwargs
    ) -> Call:
        """Create a call to the pallet function.

        Parameters:
            call_module: If not provided, will be determined from the calling class name.
            call_function: If not provided, will be determined from the calling method name.
            **kwargs: Named parameters that will be passed to the function.

        Note:
            The key in kwargs must always match the parameter name in the subtensor's function.
        """
        if call_module is None:
            call_module = self.__class__.__name__

        if call_function is None:
            call_function = get_caller_name()

        return self.subtensor.compose_call(
            call_module=call_module,
            call_function=call_function,
            call_params=kwargs,
        )

    def __getattr__(self, name: str) -> Callable[..., Call]:
        """Intercept attribute access for dynamic function calls.

        This method is called when an attribute is not found through normal means. If `dynamic_function=True`, it
        returns a callable that creates a GenericCall for the requested function name.

        Parameters:
            name: The name of the attribute/method being accessed.

        Returns:
            A callable that creates a GenericCall when invoked.

        Raises:
            AttributeError: If `dynamic_function=False` and the attribute doesn't exist.
        """
        # Don't intercept special attributes or if dynamic_function is disabled
        if name.startswith("_") or not self.dynamic_function:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'."
            )

        # Check if it's a real method that exists (shouldn't happen, but safety check)
        if hasattr(type(self), name) and not name.startswith("_"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'."
            )

        def _dynamic_call(**kwargs) -> Call:
            """Dynamically create a call for the requested function.

            Parameters:
                **kwargs: Named parameters that will be passed to the pallet function.

            Returns:
                GenericCall or Awaitable[GenericCall] depending on subtensor type.
            """
            return self.create_composed_call(call_function=name, **kwargs)

        return _dynamic_call
