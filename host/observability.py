import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# New imports for logging
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry import _logs as logs

def setup_tracing():
    """
    Initializes OpenTelemetry tracing with an OTLP exporter and logging instrumentation.
    """
    # --- Traces Setup ---
    trace_provider = TracerProvider()
    trace_exporter = OTLPSpanExporter()
    trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    trace.set_tracer_provider(trace_provider)

    # --- Logs Setup ---
    log_provider = LoggerProvider()
    log_exporter = OTLPLogExporter()
    log_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    logs.set_logger_provider(log_provider)

    # Attach the OTel handler to the root logger
    otel_handler = LoggingHandler(logger_provider=log_provider)
    logging.getLogger().addHandler(otel_handler)
    
    print("OpenTelemetry tracing initialized with OTLPLogExporter and OTLPSpanExporter.")

def get_tracer(name: str) -> trace.Tracer:
    """
    Returns a tracer with the specified name.
    
    This should be called after setup_tracing() has been run.
    """
    return trace.get_tracer(name)

# Initialize tracing on import
setup_tracing() 