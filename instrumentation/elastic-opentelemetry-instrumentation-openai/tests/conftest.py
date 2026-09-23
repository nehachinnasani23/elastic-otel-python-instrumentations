# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright
# ownership. Elasticsearch B.V. licenses this file to you under
# the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import Sequence, Tuple, Union

import openai
import pytest
import yaml
from openai._base_client import BaseClient
from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.instrumentation.openai.metrics import (
    _GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
    _GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS,
)
from opentelemetry.metrics import Histogram
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.test.globals_test import reset_metrics_globals

from .utils import assert_metric_expected, create_histogram_data_point


@pytest.fixture(scope="session")
def trace_exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    return exporter


# cannot use a fixture session scoped because we don't have a way to clear it
# TODO: expose api to reset it upstream?
@pytest.fixture
def metrics_reader():
    reset_metrics_globals()
    memory_reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[memory_reader])
    metrics.set_meter_provider(provider)

    return memory_reader


@pytest.fixture(scope="session")
def logs_exporter():
    exporter = InMemoryLogExporter()
    logger_provider = LoggerProvider()
    set_logger_provider(logger_provider)

    logger_provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))

    return exporter


@pytest.fixture(autouse=True)
def clear_exporter(trace_exporter, metrics_reader, logs_exporter):
    trace_exporter.clear()
    logs_exporter.clear()


@pytest.fixture
def instrument(monkeypatch, clear_exporter):
    monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False)
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument()

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture
def capture_message_content_instrument(monkeypatch, clear_exporter):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument()

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture
def otel_sdk_disabled_instrument(monkeypatch, clear_exporter):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument()

    yield instrumentor

    instrumentor.uninstrument()


@pytest.fixture
def vcr_cassette_name(request):
    """
    Strips `_async` from the test function name as they use the same data.
    """
    # Get the name of the test function
    test_name = request.node.name

    # Remove '_async' from the test name
    cassette_name = test_name.replace("_async", "")

    return cassette_name


OPENAI_API_KEY = "test_openai_api_key"
OPENAI_ORG_ID = "test_openai_org_id"
OPENAI_PROJECT_ID = "test_openai_project_id"


@pytest.fixture
def default_openai_env(monkeypatch):
    """
    This fixture prevents OpenAIProvider.from_env() from erring on missing
    environment variables.

    When running VCR tests for the first time or after deleting a cassette
    recording, set required environment variables, so that real requests don't
    fail. Subsequent runs use the recorded data, so don't need them.
    """
    if "OPENAI_API_KEY" not in os.environ:
        monkeypatch.setenv("OPENAI_API_KEY", OPENAI_API_KEY)


@pytest.fixture(scope="module")
def vcr_config():
    """
    This scrubs sensitive data and gunzips bodies when in recording mode.

    Without this, you would leak cookies and auth tokens in the cassettes.
    Also, depending on the request, some responses would be binary encoded
    while others plain json. This ensures all bodies are human-readable.
    """
    return {
        "decode_compressed_response": True,
        "filter_headers": [
            ("authorization", "Bearer " + OPENAI_API_KEY),
            ("openai-organization", OPENAI_ORG_ID),
            ("openai-project", OPENAI_PROJECT_ID),
            ("cookie", "test_cookie"),
        ],
        "before_record_response": scrub_response_headers,
    }


def scrub_response_headers(response):
    """
    This scrubs sensitive response headers. Note they are case-sensitive!
    """
    response["headers"]["openai-organization"] = OPENAI_ORG_ID
    response["headers"]["Set-Cookie"] = "test_set_cookie"
    return response


def address_and_port(client: BaseClient) -> Tuple[str, int]:
    url = client.base_url
    return url.host, url.port if url.port else 443


def get_integration_client():
    if "AZURE_OPENAI_ENDPOINT" in os.environ:
        return openai.AzureOpenAI()
    return openai.OpenAI()


def get_integration_async_client():
    if "AZURE_OPENAI_ENDPOINT" in os.environ:
        return openai.AsyncAzureOpenAI()
    return openai.AsyncOpenAI()


def assert_operation_duration_metric(
    client: BaseClient,
    operation_name,
    metric: Histogram,
    attributes: dict,
    min_data_point: float,
    max_data_point: float = None,
    sum_data_point: float = None,
    count: int = 1,
):
    assert metric.name == "gen_ai.client.operation.duration"
    address, port = address_and_port(client)
    default_attributes = {
        "gen_ai.operation.name": operation_name,
        "gen_ai.system": "openai",
        "server.address": address,
        "server.port": port,
    }
    # handle the simple cases of 1 or 2 data points in the histogram with just min_data_point
    if max_data_point is None:
        max_data_point = min_data_point
    if sum_data_point is None:
        if count == 1:
            sum_data_point = min_data_point
        elif count == 2:
            sum_data_point = min_data_point + max_data_point
        else:
            raise ValueError("Missing sum_data_point with more than 2 values")
    assert_metric_expected(
        metric,
        [
            create_histogram_data_point(
                count=count,
                sum_data_point=sum_data_point,
                max_data_point=max_data_point,
                min_data_point=min_data_point,
                attributes={**default_attributes, **attributes},
                explicit_bounds=tuple(_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS),
            ),
        ],
        est_value_delta=0.25,
    )


def assert_error_operation_duration_metric(
    operation_name, metric: Histogram, attributes: dict, data_point: float, value_delta: float = 0.5
):
    assert metric.name == "gen_ai.client.operation.duration"
    default_attributes = {
        "gen_ai.operation.name": operation_name,
        "gen_ai.system": "openai",
        "error.type": "APIConnectionError",
        "server.address": "localhost",
        "server.port": 9999,
    }
    assert_metric_expected(
        metric,
        [
            create_histogram_data_point(
                count=1,
                sum_data_point=data_point,
                max_data_point=data_point,
                min_data_point=data_point,
                attributes={**default_attributes, **attributes},
                explicit_bounds=tuple(_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS),
            ),
        ],
        est_value_delta=value_delta,
    )


def assert_token_usage_input_metric(
    client: BaseClient, operation_name, metric: Histogram, attributes: dict, input_data_point: int, count: int = 1
):
    assert metric.name == "gen_ai.client.token.usage"
    address, port = address_and_port(client)
    default_attributes = {
        "gen_ai.operation.name": operation_name,
        "gen_ai.system": "openai",
        "server.address": address,
        "server.port": port,
        "gen_ai.token.type": "input",
    }
    assert_metric_expected(
        metric,
        [
            create_histogram_data_point(
                count=count,
                sum_data_point=input_data_point,
                max_data_point=input_data_point,
                min_data_point=input_data_point,
                attributes={**default_attributes, **attributes},
                explicit_bounds=tuple(_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS),
            ),
        ],
    )


def assert_token_usage_metric(
    client: BaseClient,
    operation_name,
    metric: Histogram,
    attributes: dict,
    input_data_point: Union[int, Sequence[int]],
    output_data_point: Union[int, Sequence[int]],
    count: int = 1,
):
    assert metric.name == "gen_ai.client.token.usage"
    address, port = address_and_port(client)
    default_attributes = {
        "gen_ai.operation.name": operation_name,
        "gen_ai.system": "openai",
        "server.address": address,
        "server.port": port,
        "gen_ai.token.type": "input",
    }

    if count == 1:
        assert isinstance(input_data_point, int)
        assert isinstance(output_data_point, int)
        sum_input = min_input = max_input = input_data_point
        sum_output = min_output = max_output = output_data_point
    else:
        assert isinstance(input_data_point, Sequence)
        assert isinstance(output_data_point, Sequence)
        sum_input = sum(input_data_point)
        min_input = min(input_data_point)
        max_input = max(input_data_point)
        sum_output = sum(output_data_point)
        min_output = min(output_data_point)
        max_output = max(output_data_point)

    assert_metric_expected(
        metric,
        [
            create_histogram_data_point(
                count=count,
                sum_data_point=sum_input,
                max_data_point=max_input,
                min_data_point=min_input,
                attributes={**default_attributes, **attributes, "gen_ai.token.type": "input"},
                explicit_bounds=tuple(_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS),
            ),
            create_histogram_data_point(
                count=count,
                sum_data_point=sum_output,
                max_data_point=max_output,
                min_data_point=min_output,
                attributes={**default_attributes, **attributes, "gen_ai.token.type": "output"},
                explicit_bounds=tuple(_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS),
            ),
        ],
    )


def pytest_addoption(parser):
    parser.addoption(
        "--integration-tests",
        action="store_true",
        default=False,
        help="run integrations tests doing real requests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark integration tests")


def pytest_collection_modifyitems(config, items):
    run_integration_tests = bool(config.getoption("integration_tests"))
    reason = "running integrations tests only" if run_integration_tests else "skipping integration tests"
    skip_mark = pytest.mark.skip(reason=reason)
    for item in items:
        test_is_integration = "integration" in item.keywords
        if run_integration_tests != test_is_integration:
            item.add_marker(skip_mark)


class LiteralBlockScalar(str):
    """Formats the string as a literal block scalar, preserving whitespace and
    without interpreting escape characters"""


def literal_block_scalar_presenter(dumper, data):
    """Represents a scalar string as a literal block, via '|' syntax"""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(LiteralBlockScalar, literal_block_scalar_presenter)


def process_string_value(string_value):
    """Pretty-prints JSON or returns long strings as a LiteralBlockScalar"""
    try:
        json_data = json.loads(string_value)
        return LiteralBlockScalar(json.dumps(json_data, indent=2))
    except (ValueError, TypeError):
        if len(string_value) > 80:
            return LiteralBlockScalar(string_value)
    return string_value


def convert_body_to_literal(data):
    """Searches the data for body strings, attempting to pretty-print JSON"""
    if isinstance(data, dict):
        for key, value in data.items():
            # Handle response body case (e.g., response.body.string)
            if key == "body" and isinstance(value, dict) and "string" in value:
                value["string"] = process_string_value(value["string"])

            # Handle request body case (e.g., request.body)
            elif key == "body" and isinstance(value, str):
                data[key] = process_string_value(value)

            else:
                convert_body_to_literal(value)

    elif isinstance(data, list):
        for idx, choice in enumerate(data):
            data[idx] = convert_body_to_literal(choice)

    return data


class PrettyPrintJSONBody:
    """This makes request and response body recordings more readable."""

    @staticmethod
    def serialize(cassette_dict):
        cassette_dict = convert_body_to_literal(cassette_dict)
        return yaml.dump(cassette_dict, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def deserialize(cassette_string):
        return yaml.load(cassette_string, Loader=yaml.Loader)


@pytest.fixture(scope="module", autouse=True)
def fixture_vcr(vcr):
    vcr.register_serializer("yaml", PrettyPrintJSONBody)
    return vcr
