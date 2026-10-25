"""Regression checks for chronological alignment and safe offline exports."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / 'scripts/agent_viewer/export.py'
spec = importlib.util.spec_from_file_location('viewer_export', MODULE)
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


class ViewerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sandbox = self.root / 'sandbox'
        self.sandbox.mkdir()
        (self.sandbox / 'before.jpg').write_bytes(b'\xff\xd8\xffbefore')
        (self.sandbox / 'reset.jpg').write_bytes(b'\xff\xd8\xffreset')
        (self.root / 'session.json').write_text(json.dumps({'sandbox': str(self.sandbox)}))
        self.rows = [
            {'ev': 'start', 't': 0, 'model': 'example'},
            {'ev': 'user', 't': 1, 'text': 'Task', 'images': ['before.jpg']},
            {'ev': 'assistant', 'turn': 0, 't': 2, 'tool_calls': [
                {'id': 'a', 'name': 'robot', 'arguments': {'command': 'done'}}]},
            {'ev': 'tool', 'turn': 0, 't': 3, 'command': 'done', 'output': json.dumps({
                'ok': True, 'message': 'Attempt 2 of 3 starts now.'}), 'images': ['reset.jpg']},
            {'ev': 'api_error', 'turn': 1, 't': 4, 'error': 'temporary failure'},
            {'ev': 'assistant', 'turn': 1, 't': 5, 'text': 'retry', 'tool_calls': [
                {'id': 'b', 'name': 'robot', 'arguments': {'command': 'move_to'}}]},
            {'ev': 'tool', 'turn': 1, 't': 6, 'command': 'move_to', 'output': '{"ok":true}', 'images': []},
            {'ev': 'assistant', 'turn': 2, 't': 7, 'text': 'report', 'tool_calls': []},
        ]
        self.events = [{'ev': 'session_start', 'task': 'test', 'seed': 1},
                       {'ev': 'done', 'attempt': 1, 'success': False, 't': 2.5, 'feedback': 'failed'}]

    def write(self):
        for filename, rows in [('transcript.jsonl', self.rows), ('events.jsonl', self.events)]:
            (self.root / filename).write_text('\n'.join(json.dumps(r) for r in rows))

    def build(self):
        self.write()
        return viewer.build_run(self.root, {})

    def test_reset_does_not_replace_decision_input(self):
        data = self.build()
        first, second, report = data['steps']
        self.assertEqual(first['before'][0]['path'], 'before.jpg')
        self.assertEqual(first['actions'][0]['after'][0]['path'], 'reset.jpg')
        self.assertEqual(first['actions'][0]['resetToAttempt'], 2)
        self.assertFalse(first['verdict']['success'])
        self.assertEqual(second['attempt'], 2)
        self.assertEqual(second['before'][0]['path'], 'reset.jpg')
        self.assertEqual(report['before'][0]['path'], 'reset.jpg')
        self.assertEqual(second['notices'][0]['kind'], 'api_error')
        self.assertEqual(first['notices'], [])

    def test_missing_image_and_unfinished_call_are_explicit(self):
        (self.sandbox / 'before.jpg').unlink()
        self.rows = self.rows[:3]
        data = self.build()
        self.assertIsNone(data['steps'][0]['before'][0]['asset'])
        self.assertTrue(data['steps'][0]['incomplete'])
        self.assertEqual(data['status'], 'incomplete')
        self.assertEqual(len(data['warnings']), 1)

    def test_recorded_summary_is_safely_embedded_as_data(self):
        self.rows[2]['text'] = '</script><script>alert(1)</script>'
        self.rows[2]['reasoning_summary'] = '</script><script>SUMMARY_SENTINEL</script>'
        self.write()
        out = self.root / 'viewer.html'
        viewer.render([self.root], out)
        html = out.read_text()
        self.assertNotIn('</script><script>alert(1)', html)
        self.assertIn('SUMMARY_SENTINEL', html)
        self.assertNotIn('</script><script>SUMMARY_SENTINEL', html)
        self.assertIn('\\u003c/script\\u003e', html)

    def test_summary_belongs_to_the_next_decision_not_previous_action(self):
        self.rows[5]['reasoning_summary'] = 'Adjust after feedback'
        data = self.build()
        self.assertIsNone(data['steps'][0]['decisionSummary'])
        self.assertEqual(data['steps'][1]['decisionSummary'], 'Adjust after feedback')
        self.assertEqual(data['steps'][1]['decisionSourceLine'], 6)
        self.assertEqual(data['steps'][1]['turn'], 1)

    def test_mismatched_tool_turn_fails_instead_of_silent_alignment(self):
        self.rows[3]['turn'] = 999
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.build()

    def test_call_ids_handle_reordered_parallel_results(self):
        self.rows[2]['tool_calls'].append({'id': 'other', 'name': 'robot', 'arguments': {'command': 'status'}})
        self.rows.insert(3, {'ev': 'tool', 'turn': 0, 't': 2.2, 'call_id': 'other', 'output': 'other result'})
        first = self.build()['steps'][0]
        self.assertEqual(first['actions'][1]['result'], 'other result')
        self.assertIn('Attempt 2', first['actions'][0]['result'])

    def test_malformed_json_is_not_silently_dropped(self):
        self.write()
        (self.root / 'transcript.jsonl').write_text('{broken')
        with self.assertRaisesRegex(ValueError, 'malformed JSON'):
            viewer.build_run(self.root, {})

    def test_paths_cannot_escape_selected_image_roots(self):
        self.rows[1]['images'] = ['/etc/passwd']
        data = self.build()
        self.assertIsNone(data['steps'][0]['before'][0]['asset'])


if __name__ == '__main__':
    unittest.main()
