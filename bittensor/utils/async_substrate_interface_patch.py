import weakref
from async_substrate_interface.utils import cache


class WeakMethodCallable:
    """
    A callable that holds a weak reference to a bound method.
    Used to break reference cycles in CachedFetcher.
    """

    def __init__(self, bound_method):
        self._weak_method = weakref.WeakMethod(bound_method)

    async def __call__(self, *args, **kwargs):
        method = self._weak_method()
        if method is None:
            # The underlying method/instance has been garbage collected.
            # Return None gracefully instead of raising, so callers of
            # CachedFetcher do not see a low-level ReferenceError.
            return None
        return await method(*args, **kwargs)


def _new_get(self, instance, owner):
    """
    Patched __get__ method for _CachedFetcherMethod to use WeakKeyDictionary
    and WeakMethodCallable, preventing memory leaks.
    """
    if instance is None:
        return self

    # Migration/Safety: Ensure _instances is a WeakKeyDictionary.
    # If it's the original dict (from before patching), replace it,
    # preserving any existing cached instances.
    if not isinstance(self._instances, weakref.WeakKeyDictionary):
        self._instances = weakref.WeakKeyDictionary(self._instances)

    # Cache per-instance
    if instance not in self._instances:
        bound_method = self.method.__get__(instance, owner)

        # Use WeakMethodCallable to avoid a strong reference cycle that would otherwise be:
        # _CachedFetcherMethod (class attr) -> WeakKeyDictionary -> CachedFetcher -> bound method -> instance.
        # WeakMethodCallable stores a weakref.WeakMethod instead of the bound method itself, so it does not
        # keep a strong reference to the instance and thus breaks this potential cycle.
        wrapped_method = WeakMethodCallable(bound_method)

        self._instances[instance] = cache.CachedFetcher(
            max_size=self.max_size,
            method=wrapped_method,
            cache_key_index=self.cache_key_index,
        )
    return self._instances[instance]


def apply_patch():
    """
    Applies the patch to async_substrate_interface.utils.cache._CachedFetcherMethod.
    """
    target_class = cache._CachedFetcherMethod
    # Check if already patched to avoid recursion or double patching issues
    if getattr(target_class, "_bittensor_patched", False):
        return

    target_class.__get__ = _new_get
    target_class._bittensor_patched = True
