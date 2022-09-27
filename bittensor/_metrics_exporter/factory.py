from bittensor._metrics_exporter.prometheus_exporter import PrometheusExporter


METRICS_EXPORTER_PROMETHEUS = 'prometheus'


class MetricsExporterFactory:

    def get_metrics_exporter(self, metrics_exporter = METRICS_EXPORTER_PROMETHEUS):
        if METRICS_EXPORTER_PROMETHEUS == metrics_exporter:
            return PrometheusExporter()

        raise ValueError(metrics_exporter)
