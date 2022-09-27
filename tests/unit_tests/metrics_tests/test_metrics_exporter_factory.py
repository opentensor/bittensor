import pytest

from bittensor._metrics_exporter.factory import MetricsExporterFactory, METRICS_EXPORTER_PROMETHEUS
from bittensor._metrics_exporter.prometheus_exporter import PrometheusExporter


def test_metrics_exporter_factory_when_unexpected_value():
    # given:
    factory = MetricsExporterFactory()

    # when: + then:
    with pytest.raises(ValueError):
        factory.get_metrics_exporter('hahaha')


def test_metrics_exporter_factory_when_no_value_then_prometheus_exporter():
    # given:
    factory = MetricsExporterFactory()
    expected_class = PrometheusExporter.__name__

    # when:
    metrics_exporter = factory.get_metrics_exporter()

    # then:
    obtained_class = type(metrics_exporter).__name__
    assert expected_class == obtained_class


def test_metrics_exporter_factory_when_prometheus_value_then_prometheus_exporter():
    # given:
    factory = MetricsExporterFactory()
    expected_class = PrometheusExporter.__name__

    # when:
    metrics_exporter = factory.get_metrics_exporter(METRICS_EXPORTER_PROMETHEUS)

    # then:
    obtained_class = type(metrics_exporter).__name__
    assert expected_class == obtained_class
