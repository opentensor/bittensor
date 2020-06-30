import opentensor_proto
import numpy


def test_serialize():
    for _ in range(10):
        array = numpy.random.rand(10,10)
        content = opentensor_proto.serialize(array)
        array_d = opentensor_proto.deserialize(content)
        numpy.testing.assert_array_equal(array, array_d)
