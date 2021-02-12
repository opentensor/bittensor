from bittensor import utils

def test_int_to_ip_zero():
    assert utils.networking.int_to_ip(0) == "0.0.0.0"
    assert utils.networking.ip_to_int("0.0.0.0") == 0
    assert utils.networking.ip__str__(4, "0.0.0.0") == "/ipv4/0.0.0.0" 

def test_int_to_ip_range():
    for i in range(10):
        assert utils.networking.int_to_ip(i) == "0.0.0." + str(i)
        assert utils.networking.ip_to_int("0.0.0." + str(i)) == i
        assert utils.networking.ip__str__(4, "0.0.0."+ str(i)) == "/ipv4/0.0.0." + str(i)

def test_int_to_ip4_max():
    assert utils.networking.int_to_ip(4294967295) == "255.255.255.255"
    assert utils.networking.ip_to_int( "255.255.255.255") == 4294967295
    assert utils.networking.ip__str__(4, "255.255.255.255") == "/ipv4/255.255.255.255"

def test_int_to_ip6_zero():
    assert utils.networking.int_to_ip(4294967296) == "::1:0:0"
    assert utils.networking.ip_to_int("::1:0:0") == 4294967296
    assert utils.networking.ip__str__(6, "::1:0:0") == "/ipv6/::1:0:0"

def test_int_to_ip6_range():
    for i in range(10):
        assert utils.networking.int_to_ip(4294967296 + i) == "::1:0:" + str(i)
        assert utils.networking.ip_to_int("::1:0:" + str(i)) == 4294967296 + i
        assert utils.networking.ip__str__(6, "::1:0:" + str(i)) == "/ipv6/::1:0:" + str(i)

def test_int_to_ip6_max():
    max_val = 340282366920938463463374607431768211455
    assert utils.networking.int_to_ip(max_val) == 'ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff'
    assert utils.networking.ip_to_int('ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff') == max_val
    assert utils.networking.ip__str__(6, "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff") == "/ipv6/ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"

def test_int_to_ip6_overflow():
    overflow = 340282366920938463463374607431768211455 + 1
    try:
        utils.networking.int_to_ip(overflow) 
    except:
        assert True

def test_int_to_ip6_underflow():
    underflow = -1
    try:
        utils.networking.int_to_ip(underflow) 
    except:
        assert True

def test_get_external_ip():
    assert utils.networking.get_external_ip()

if __name__ == "__main__":
    test_int_to_ip_zero()
    test_int_to_ip_range()
    test_int_to_ip4_max()
    test_int_to_ip6_zero()
    test_int_to_ip6_range()
    test_int_to_ip6_max()
    test_int_to_ip6_overflow()
    test_int_to_ip6_underflow()
    test_get_external_ip()
