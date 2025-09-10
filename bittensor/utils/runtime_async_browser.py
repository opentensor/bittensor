import asyncio
import bittensor as bt
from .btlogging import logging

# Default network in case user does not explicitly initialize runtime browser.
DEFAULT_NETWORK='local'

TYPE='ty' # depending on where we get metadata from, we have 'ty' or 'type', TODO: settle on metadata source

class runtime_async_browser_imp(type):
    async_subtensor = None
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

    def init(cls,network=None,async_subtensor=None,block=None):
        cls.block = block
        if async_subtensor:
            if async_subtensor != cls.async_subtensor:
                cls.async_subtensor = async_subtensor
                cls.network = None
                cls.items.clear()
        else:
            if not network:
                logging.warning(f"Initializing runtime browser with default network {DEFAULT_NETWORK}, block={block}.")
                network = DEFAULT_NETWORK
            if network != cls.network:
                cls.items.clear()
                cls.network = network
                cls.async_subtensor = bt.async_subtensor(network=network)

    def get_async_subtensor(cls):
        if cls.async_subtensor is None:
            cls.init()
        return cls.async_subtensor

    async def init_metadata(cls):
        async_subtensor = cls.get_async_subtensor()
        async_substrate = async_subtensor.substrate
        if cls.block:
            block = cls.block
        else:
            block = await async_subtensor.block
        block_hash = await async_substrate.get_block_hash(block)
        rt = await async_substrate.init_runtime(block_hash=block_hash)
        if 1:
            cls.subtensor_metadata = rt.metadata_v15.value()
        else:
            # not sure if this is the most future proof way to go to get to the metadata
            md_dict = substrate.metadata.value[1]
            version = list(md_dict.keys())[0]
            cls.subtensor_metadata = md_dict[version]
            cls.subtensor_metadata['apis'] = rt.metadata_v15.value()['apis']

    # blocking at first call
    def get_metadata(cls):
        if cls.subtensor_metadata is None:
            raise Exception(f'Metadata not initialized; call await bittensor.async_runtime.init_metadata() before using async_runtime')
        return cls.subtensor_metadata

    # blocking at first call
    def get_pallets(cls):
        if cls.pallets is None:
            md = cls.get_metadata()
            cls.pallets = {p.get('name','?'):p for p in md['pallets']}
        return cls.pallets

    # blocking at first call
    def get_apis(cls):
        if cls.apis is None:
            #cls.apis = list(bt.core.settings.TYPE_REGISTRY['runtime_api'].keys())
            md = cls.get_metadata()
            cls.apis = [item['name'] for item in md['apis']]
        return cls.apis

    def dir(cls):
        pallets = cls.get_pallets()
        apis = cls.get_apis()
        return list(pallets.keys())+apis

    # We cannot make __getattr__ async, or the end user would have to await the
    # module _and_ the item within the module, that is, we don't want this:
    # await (await bt.async_runtime.SubtensorModule).NetworkLastRegistered()
    # So necessarily the first invocation will be blocking, as it needs to
    # fetch metadata.
    def __getattr__(cls,item):
        if item == '__super__':
            return object
        if not item[0].isalpha():
            try:
                ret = cls.__super__.__getattr__(item)
            except Exception as e:
                raise
            return ret
        if item == 'metadata':
            return cls.subtensor_metadata
        if item in cls.get_pallets():
            if not item in cls.items:
                cls.items[item] = subbrowser_module(
                        subbrowser=cls,
                        module=item,
                        metadata=cls.get_pallets()[item],
                    )
            return cls.items[item]
        apis = {}
        try:
            apis = cls.get_apis()
        except Exception as e:
            logging.warning(f'Error fetching apis - this is an internal bittensor.runtime error')

        if item in apis:
            if not item in cls.items:
                cls.items[item] = subbrowser_api(subbrowser=cls,api=item)
        else:
            raise Exception(f"Pallet or API {item} not found; use bittensor.runtime.dir() to get valid options")
        return cls.items[item]

    def __repr__(cls):
        return f'<dynamic interface to bittensor runtime>'

# this makes bittensor.runtime behave like an object, without prior instantiation
class runtime_async_browser(metaclass=runtime_async_browser_imp):
    def __dir__():
        return bt.runtime.dir()

# proxy for subtensor module, e.g. bt.runtime.SubtensorModule or bt.runtime.System
class subbrowser_module:
    _subbrowser = None
    _module = None
    _objs = None
    _storage_entries = None
    _constants = None
    _metadata = None

    def __init__(self,subbrowser=None,module=None,metadata=None):
        self._subbrowser = subbrowser
        self._module = module
        self._objs = {}
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
                msg += '\nPossibly the entry exists on a different block height; explicitly initialize on a block using e.g. bt.runtime.init(block=4000000)'
                # TODO: implement a delayed error object, so that you can call bt.runtime.SubtensorModule.SubnetLimit(block=4000000) after SubnetLimit does not exist anymore?
                raise Exception(msg)
            # It is a design decision to not return the value of plain types, but still return an
            # object, that can be queried e.g. like bt.runtime.Timestamp.Now(), because it keeps
            # the option open to add block=... kwarg.
            # The alternative is to test 'Plain' in md[TYPE] and perform the query here.
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
        if type(self._metadata[TYPE]) is not dict:
            return 0
        if 'Plain' in self._metadata[TYPE]:
            return 0
        if 'Map' in self._metadata[TYPE]:
            mapinfo = self._metadata[TYPE]['Map']
            hashers = mapinfo.get('hashers',[])
            return len(hashers)

    async def _query(self,*args,**kwargs):
        if len(args) != self._n_indices:
            if self._n_indices == 0:
                raise Exception(f'plain item {self._fullname} does not accept indices; use {self._fullname}()')
            raise Exception(f'map {self._fullname} requires {self._n_indices} indices')
        if type(self._metadata[TYPE]) is not dict:
            pass
        elif 'Plain' in self._metadata[TYPE]:
            pass
        elif 'Map' in self._metadata[TYPE]:
            # Specifically ensure that we enfore ss58 indices, showing pretty errors.
            mapinfo = self._metadata[TYPE]['Map']
            hashers = mapinfo.get('hashers',[])
            for i,h in enumerate(hashers):
                if h in ('Blake2_128Concat'): # in Stake "Identity" is an ss58 addr, in Alpha and other maps, "Identity" is an integer.
                    if type(args[i]) != str or len(args[i]) != 48:
                        raise Exception(f'index {i} of {self._fullname} should be an ss58 address')
        if self._type == 'storage':
            ret = await self._subbrowser_module._subbrowser.async_subtensor.query_module(
                    module=self._subbrowser_module._module,
                    name=self._name,
                    params=args,
                    block=kwargs.get('block',self._subbrowser_module._subbrowser.block)
                )
            # don't even try to understand the logic of what query_module() returns; might be scale obj, str or int
            try:
                return ret.value
            except:
                return ret
        elif self._type == 'constant':
            ret = await self._subbrowser_module._subbrowser.async_subtensor.query_constant(
                    module_name=self._subbrowser_module._module,
                    constant_name=self._name,
                    block=kwargs.get('block',self._subbrowser_module._subbrowser.block)
                )
            try:
                return ret.value
            except:
                return ret

    async def __call__(self,*args,**kwargs):
        #logging.warning(f'objproxy call like access: {args} // {kwargs}')
        return await self._query(*args,**kwargs)

    def __getitem__(self,args):
        logging.warning(f'indexed querying of runtime objects is officially not async, this may break in the future; use object(index0,...) instead of object[index0,...]')
        # TODO: test if it might still work anyway to return a coroutine; we're not performing del or assigning values anyway
        if type(args) != tuple: args = (args,)
        return self._query(*args)

    def __repr__(self):
        if self._n_indices>0:
            indices = ','.join(f'index{i}' for i in range(self._n_indices))
            return f'<dynamic interface to map {self._fullname} in the bittensor runtime; add [{indices}] to query>'
        return f'<dynamic interface to {self._type} value {self._fullname} in the bittensor runtime; add () to query>'

# proxy for subtensor api, e.g. in order to call:
#  bt.runtime.SubnetRegistrationRuntimeApi.get_network_registration_cost()
#  bt.runtime.SubnetRegistrationRuntimeApi.get_network_registration_cost(block=12345)
class subbrowser_api:
    _subbrowser = None
    _api = None
    _calls = None
    def __init__(self,subbrowser=None,api=None):
        self._subbrowser = subbrowser
        self._api = api
        self._calls = {}
        self._methods = None
        md = self._subbrowser.get_metadata()
        for item in md['apis']:
            if item['name'] == api:
                self._methods = [m['name'] for m in item['methods']]
        if self._methods is None:
            logging.warning(f'Failed to get methods for api {api} from metadata')
            self._methods = []

    def dir(self):
        #return list(bt.core.settings.TYPE_REGISTRY['runtime_api'][self._api]["methods"].keys())
        return self._methods

    def __getattr__(self,call):
        if not call in self._calls:
            #methods = bt.core.settings.TYPE_REGISTRY["runtime_api"][self._api]["methods"]
            if not call in self._methods:
                raise Exception(f'API call {self._api}.{call} not found; available calls are {", ".join(self._methods)}')
            async def query(*args,block=None):
                return await self._subbrowser.async_subtensor.query_runtime_api(
                        self._api,
                        call,
                        args,
                        block=block or self._subbrowser.block,
                    )
            self._calls[call] = query
        return self._calls[call]

    def __repr__(self):
        return f'<dynamic interface to api {self._api} in the bittensor runtime>'
