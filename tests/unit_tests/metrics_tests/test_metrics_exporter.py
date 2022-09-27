import pytest

from bittensor._metrics_exporter.factory import METRICS_EXPORTER_PROMETHEUS
from bittensor._metrics_exporter import metrics_exporter


class PrometheusExporterMock:
    pass


class MetricsExporterFactoryMock:
    def get_metrics_exporter(self, value = METRICS_EXPORTER_PROMETHEUS):
        if METRICS_EXPORTER_PROMETHEUS == value:
            return PrometheusExporterMock()

        raise ValueError(value)


class MetricsExporterSUT(metrics_exporter, MetricsExporterFactoryMock):
    '''
    In this way we are injecting the mock class into metrics_exporter
    So, by taking advantage of the Python MRO, the mock is going to be called first.
    '''


def test_metrics_exporter_singleton_when_creating_many_times():
   # given:

   # when:
   sut1 = MetricsExporterSUT()
   sut2 = MetricsExporterSUT()

   # then:
   assert sut1 == sut2
