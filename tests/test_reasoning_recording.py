"""Summary opt-in and transcript persistence without model or simulator calls."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from lffbench.agent.context import Conversation
from lffbench.agent.backends.base import BackendResponse
from lffbench.agent.harness import AgentHarness

spec = importlib.util.spec_from_file_location('run_agent_script', Path(__file__).parents[1] / 'scripts/run_agent.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_default_summary_request_and_explicit_disable(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-test')
    for flags, expected in [([], 'auto'), (['--reasoning-summary', 'none'], None)]:
        args = runner.parse(['--task', 'test', '--backend', 'openai'] + flags)
        backend = runner.build_backend(args)
        request = backend.build_request(Conversation('test'), [])
        assert request['reasoning'].get('summary') == expected


@pytest.mark.parametrize('summary,status', [('Adjust after feedback', 'recorded'), (None, 'not_returned')])
def test_summary_and_public_text_persist_with_availability(summary, status):
    harness = SimpleNamespace(cfg=SimpleNamespace(model_for_cost='none'), payload_probe=None, log=Mock())
    response = BackendResponse(reasoning_summary=summary, text='Moving 4 cm less', latency_s=1.2)
    record = AgentHarness._log_llm_call(harness, 1000, 2, 9, response, Conversation('test'), [])
    assert record['reasoning_summary'] == summary
    assert record['reasoning_summary_status'] == status
    assert record['text'] == 'Moving 4 cm less'
    harness.log.event.assert_called_once_with('llm_call', **record)
