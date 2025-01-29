import bittensor as bt
from .btlogging import logging

# Default network in case user does not explicitly initialize runtime browser.
DEFAULT_NETWORK='local'

class runtime_browser_imp(type):
    subtensor = None
    pallets = None
    apis = None
    subtensor_metadata = None
    block = None # block used to initialize runtime and get metadata
    network = None
    items = {}

    # Support init through bt.runtime(network=...,block=...)
    def __call__(cls,**kwargs):
        cls.init(**kwargs)
        return cls

    def init(cls,network=None,subtensor=None,block=None):
        cls.block = block
        if subtensor:
            if subtensor != cls.subtensor:
                cls.subtensor = subtensor
                cls.network = None
                cls.items.clear()
        else:
            if not network:
                logging.warning(f"Initializing runtime browser with default network {DEFAULT_NETWORK}.")
                network = DEFAULT_NETWORK
            if network != cls.network:
                cls.items.clear()
                cls.network = network
                cls.subtensor = bt.subtensor(network=network)

    def get_subtensor(cls):
        if cls.subtensor is None:
            cls.init()
        return cls.subtensor

    def get_metadata(cls):
        if cls.subtensor_metadata is None:
            subtensor = cls.get_subtensor()
            substrate = subtensor.substrate
            block = cls.block or subtensor.block
            block_hash = substrate.get_block_hash(block)
            substrate.init_runtime(block_hash=block_hash)
            # not sure if this is the most future proof way to go to get to the metadata
            md_dict = substrate.metadata.value[1]
            version = list(md_dict.keys())[0]
            cls.subtensor_metadata = md_dict[version]
        return cls.subtensor_metadata

    def get_pallets(cls):
        if cls.pallets is None:
            md = cls.get_metadata()
            cls.pallets = {p.get('name','?'):p for p in md['pallets']}
        return cls.pallets

    def get_apis(cls):
        if cls.apis is None:
            cls.apis = list(bt.core.settings.TYPE_REGISTRY['runtime_api'].keys())
        return cls.apis

    def dir(cls):
        return list(cls.get_pallets().keys())+cls.get_apis()

    def __getattr__(cls,item):
        if item == '__super__':
            return object
        if not item[0].isalpha():
            try:
                ret = cls.__super__.__getattr__(item)
            except Exception as e:
                raise
            return ret
        if item in cls.get_pallets():
            if not item in cls.items:
                cls.items[item] = subbrowser_module(
                        subbrowser=cls,
                        module=item,
                        metadata=cls.get_pallets()[item],
                    )
        elif item in cls.get_apis():
            if not item in cls.items:
                cls.items[item] = subbrowser_api(subbrowser=cls,api=item)
        else:
            raise Exception(f"Pallet or API {item} not found; use bittensor.runtime.dir() to get valid options")
        return cls.items[item]

    def __repr__(cls):
        return f'<dynamic interface to bittensor runtime>'

# this makes bittensor.runtime behave like an object, without prior instantiation
class runtime_browser(metaclass=runtime_browser_imp):
    def __dir__():
        return bt.runtime.dir()

# proxy for subtensor module, e.g. bt.runtime.SubtensorModule or bt.runtime.System
class subbrowser_module:
    _subbrowser = None
    _module = None
    _objs = {}
    _storage_entries = None
    _constants = None
    _metadata = None

    def __init__(self,subbrowser=None,module=None,metadata=None):
        self._subbrowser = subbrowser
        self._module = module
        self._metadata = metadata
        self._storage_entries = {item.get('name','?'):item for item in metadata['storage']['entries']}
        self._constants = {item.get('name','?'):item for item in metadata['constants']}

    def dir(self):
        return list(self._constants.keys())+list(self._storage_entries.keys())

    def __getattr__(self,name):
        if not name in self._objs:
            md = tp = None
            if name in self._storage_entries:
                md = self._storage_entries[name]
                tp = 'storage'
            elif name in self._constants:
                md = self._constants[name]
                tp = 'constant'
            else:
                msg = f'Storage entry or constant "{name}" not found; available are: '
                msg += 'storage entries '+', '.join(self._storage_entries.keys())
                msg += ' and constants '+', '.join(self._constants.keys())
                raise Exception(msg)
            # It is a design decision to not return the value of plain types, but still return an
            # object, that can be queried e.g. like bt.runtime.Timestamp.Now(), because it keeps
            # the option open to add block=... kwarg.
            # The alternative is to test 'Plain' in md['type'] and perform the query here.
            self._objs[name] = subbrowser_objproxy(
                    subbrowser_module=self,
                    name=name,
                    metadata=md,
                    tp=tp,
                )
        return self._objs[name]

    def __repr__(self):
        return f'<dynamic interface to module {self._module} in the bittensor runtime, exposing storage entries {", ".join(self._storage_entries.keys())} and constants {", ".join(self._constants.keys())}>'

# proxy for maps and singular elements, either constants or storage entries e.g.
#  bt.runtime.Timestamp.Now (having no index)
#  bt.runtime.SubtensorModule.LastAdjustmentBlock (having 1D index: netuid)
#  bt.runtime.System.Account (having 1D index: coldkey_ss58)
#  bt.runtime.Commitments.RateLimit (constant, no index)
# accept:
#  obj()
#  obj.query()
#  obj[29]
#  obj(29)
#  obj.query(29)
# and for the call-like interfaces, kwargs:
#  obj(block=12345)
#  obj.query(block=12345)
#  obj(29,block=12345)
#  obj.query(29,block=12345)
class subbrowser_objproxy:
    _subbrowser_module = None
    _name = None
    _metadata = None
    _type = None
    _n_indices = None

    def __init__(self,subbrowser_module=None,name=None,metadata=None,tp=None):
        self._name = name
        self._subbrowser_module = subbrowser_module
        self._metadata = metadata
        self._fullname = self._subbrowser_module._module+'.'+self._name
        self._type = tp
        self._n_indices = self._get_n_indices()

    def _get_n_indices(self):
        if type(self._metadata['type']) is not dict:
            return 0
        if 'Plain' in self._metadata['type']:
            return 0
        if 'Map' in self._metadata['type']:
            mapinfo = self._metadata['type']['Map']
            hashers = mapinfo.get('hashers',[])
            return len(hashers)

    def _query(self,*args,**kwargs):
        if len(args) != self._n_indices:
            if self._n_indices == 0:
                raise Exception(f'plain item {self._fullname} does not accept indices; use {self._fullname}()')
            raise Exception(f'map {self._fullname} requires {self._n_indices} indices')
        if type(self._metadata['type']) is not dict:
            pass
        elif 'Plain' in self._metadata['type']:
            pass
        elif 'Map' in self._metadata['type']:
            # Specifically ensure that we enfore ss58 indices, showing pretty errors.
            mapinfo = self._metadata['type']['Map']
            hashers = mapinfo.get('hashers',[])
            for i,h in enumerate(hashers):
                if h in ('Blake2_128Concat'): # in Stake "Identity" is an ss58 addr, in Alpha and other maps, "Identity" is an integer.
                    if type(args[i]) != str or len(args[i]) != 48:
                        raise Exception(f'index {i} of {self._fullname} should be an ss58 address')
        if self._type == 'storage':
            return self._subbrowser_module._subbrowser.subtensor.query_module(
                    module=self._subbrowser_module._module,
                    name=self._name,
                    params=args,
                    block=kwargs.get('block',self._subbrowser_module._subbrowser.block)
                ).value
        elif self._type == 'constant':
            return self._subbrowser_module._subbrowser.subtensor.query_constant(
                    module_name=self._subbrowser_module._module,
                    constant_name=self._name,
                    block=kwargs.get('block',self._subbrowser_module._subbrowser.block)
                ).value

    def __call__(self,*args,**kwargs):
        return self._query(*args,**kwargs)

    def __getitem__(self,args):
        if type(args) != tuple: args = (args,)
        return self._query(*args)

    def __repr__(self):
        if self._n_indices>0:
            indices = ','.join(f'index{i}' for i in range(self._n_indices))
            return f'<dynamic interface to map {self._fullname} in the bittensor runtime; add [{indices}] to query>'
        return f'<dynamic interface to value {self._fullname} in the bittensor runtime; add () to query>'

# proxy for subtensor api, e.g. in order to call:
#  bt.runtime.SubnetRegistrationRuntimeApi.get_network_registration_cost()
#  bt.runtime.SubnetRegistrationRuntimeApi.get_network_registration_cost(block=12345)
class subbrowser_api:
    _subbrowser = None
    _api = None
    _calls = {}
    def __init__(self,subbrowser=None,api=None):
        self._subbrowser = subbrowser
        self._api = api

    def dir(self):
        return list(bt.core.settings.TYPE_REGISTRY['runtime_api'][self._api]["methods"].keys())

    def __getattr__(self,call):
        if not call in self._calls:
            methods = bt.core.settings.TYPE_REGISTRY["runtime_api"][self._api]["methods"]
            if not call in methods:
                raise Exception(f'API call {self._api}.{call} not found; available calls are {", ".join(methods.keys())}')
            def query(*args,block=None):
                return self._subbrowser.subtensor.query_runtime_api(
                        self._api,
                        call,
                        args,
                        block=block or self._subbrowser.block,
                    )
            self._calls[call] = query
        return self._calls[call]

    def __repr__(self):
        return f'<dynamic interface to api {self._api} in the bittensor runtime>'

