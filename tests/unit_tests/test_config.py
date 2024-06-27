import os
import sys
import pytest
import argparse
import copy
from unittest.mock import patch, mock_open
from bittensor import config


@pytest.fixture
def mock_defaults():
    class MockDefaults:
        config = {
            "path": "~/.bittensor/",
            "some_default": "default_value"
        }

    return MockDefaults


def test_init_with_no_args():
    cfg = config()
    assert cfg is not None
    assert isinstance(cfg, config)


def test_init_with_parser():
    parser = argparse.ArgumentParser()
    cfg = config(parser=parser)
    assert cfg is not None
    assert isinstance(cfg, config)


def test_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_arg', type=str, default='default_value')
    cfg = config(parser=parser, args=['--test_arg', 'new_value'])
    assert cfg.test_arg == 'new_value'


def test_parse_args_strict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_arg', type=str, default='default_value')
    with pytest.raises(SystemExit):
        cfg = config(parser=parser, args=['--unknown_arg', 'value'], strict=True)


def test_load_from_env_vars():
    with patch.dict(os.environ, {"BTCLI_TEST_VAR": "env_value"}):
        cfg = config()
        cfg.load_config_from_env_vars()
        assert cfg.env_config.get('test.var') == 'env_value'


def test_load_or_create_generic_config(mock_defaults):
    with patch("builtins.open", mock_open(read_data="some_default: 'file_value'")), \
            patch("os.path.exists", return_value=True), \
            patch("os.makedirs"):
        cfg = config()
        cfg.load_or_create_generic_config()
        assert cfg.generic_config.get('some_default') == 'file_value'


def test_load_active_profile(mock_defaults):
    with patch("builtins.open", mock_open(read_data="profile_setting: 'profile_value'")), \
            patch("os.path.exists", return_value=True), \
            patch.dict('os.environ', {'BTCLI_PROFILE_ACTIVE': 'default', 'BTCLI_PROFILE_PATH': '~/fake_path/'}):
        cfg = config()
        cfg.load_active_profile()
        assert cfg.profile_config.get('profile_setting') == 'profile_value'


def test_merge_configs():
    cfg1 = config()
    cfg2 = config()
    cfg1.test_key = "value1"
    cfg2.test_key = "value2"
    merged_cfg = config.merge_all([cfg1, cfg2])
    assert merged_cfg.test_key == "value2"


def test_is_set():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_arg', type=str)
    cfg = config(parser=parser, args=['--test_arg', 'value'])
    assert cfg.is_set('test_arg') == True
    assert cfg.is_set('unset_arg') == False


def test_check_for_missing_required_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--required_arg', type=str, required=True)
    with pytest.raises(ValueError) as e:
        cfg = config(parser=parser, args=[])
    assert "Missing required arguments: --required_arg" in str(e.value)


def test_unflatten_dict():
    flat_dict = {'a.b.c': 1, 'a.b.d': 2, 'e': 3}
    nested_dict = config.unflatten_dict(flat_dict)
    assert nested_dict == {'a': {'b': {'c': 1, 'd': 2}}, 'e': 3}


def test_remove_private_keys():
    test_dict = {
        "key1": "value1",
        "__parser": "should be removed",
        "__is_set": "should be removed",
        "nested": {
            "key2": "value2",
            "__parser": "should be removed from nested"
        }
    }
    cleaned_dict = config._remove_private_keys(test_dict)
    assert "__parser" not in cleaned_dict
    assert "__is_set" not in cleaned_dict
    assert "__parser" not in cleaned_dict["nested"]


def test_str_method():
    cfg = config()
    cfg.key1 = "value1"
    cfg.__parser = "should be removed"
    cfg.__is_set = "should be removed"
    expected_output = "\nkey1: value1\n"
    assert str(cfg) == expected_output


def test_copy_method():
    cfg = config()
    cfg.key1 = "value1"
    copied_cfg = cfg.copy()
    assert copied_cfg.key1 == "value1"
    assert copied_cfg is not cfg


def test_to_string_method():
    cfg = config()
    cfg.key1 = "value1"
    items = cfg.copy()
    items.pop('__is_set', None)
    expected_output = "\nkey1: value1\n"
    assert cfg.to_string(items) == expected_output


def test_update_with_kwargs():
    cfg = config()
    cfg.update_with_kwargs({"key1": "value1", "key2": "value2"})
    assert cfg.key1 == "value1"
    assert cfg.key2 == "value2"


def test_deepcopy_method():
    cfg = config()
    cfg.key1 = "value1"
    copied_cfg = copy.deepcopy(cfg)
    assert copied_cfg.key1 == "value1"
    assert copied_cfg is not cfg


def test_merge_static_method():
    a = {'key1': 'value1', 'key2': {'subkey1': 'subvalue1'}}
    b = {'key2': {'subkey1': 'new_subvalue1', 'subkey2': 'subvalue2'}, 'key3': 'value3'}
    merged = config._merge(a, b)
    expected = {'key1': 'value1', 'key2': {'subkey1': 'new_subvalue1', 'subkey2': 'subvalue2'}, 'key3': 'value3'}
    assert merged == expected


def test_merge_method():
    cfg1 = config()
    cfg2 = config()
    cfg1.key1 = "value1"
    cfg2.key2 = "value2"
    cfg1.merge(cfg2)
    assert cfg1.key1 == "value1"
    assert cfg1.key2 == "value2"
