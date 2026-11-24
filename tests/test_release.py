"""CPU-only checks for the portable single-sample inference release."""
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile
import unittest

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from quantwm.generate import MODELS, WEIGHT_VARIABLES, configure_cuda_linker
from quantwm.history import HistoricalQueryStatsTracker


class DocumentationAndCliTests(unittest.TestCase):
    def test_language_switches_and_local_links(self):
        for name, other in [('README.md', 'README_zh-CN.md'), ('README_zh-CN.md', 'README.md')]:
            file = PROJECT / name
            text = file.read_text()
            self.assertIn(f']({other})', text)
            self.assertEqual(text.count('```') % 2, 0)
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if target.startswith(('http:', 'https:', '#')):
                    continue
                self.assertTrue((file.parent / target.split('#')[0]).exists(), target)

    def test_all_fifteen_method_scripts_exist(self):
        for model in MODELS:
            for method in ('bf16', 'qvg', 'quantwm'):
                file = PROJECT / 'scripts' / model / (method + '.sh')
                self.assertTrue(file.is_file(), file)
                subprocess.run(['bash', '-n', str(file)], check=True)
                self.assertIn(model, file.read_text())
                self.assertIn(method, file.read_text())

    def test_portable_configuration_template(self):
        text = (PROJECT / 'configs/paths.env.example').read_text()
        self.assertNotIn('/root/', text)
        self.assertNotIn('/share/', text)
        for variables in WEIGHT_VARIABLES.values():
            for name in variables:
                self.assertIn('export ' + name, text)
        subprocess.run(['bash', '-n', str(PROJECT / 'configs/paths.env.example')], check=True)

    def test_missing_configuration_has_actionable_error(self):
        env = os.environ.copy()
        for variables in WEIGHT_VARIABLES.values():
            for name in variables:
                env.pop(name, None)
        result = subprocess.run([sys.executable, '-B', str(PROJECT / 'quantwm/generate.py'),
            '--model', 'matrix_game2', '--method', 'quantwm', '--dry-run'],
            env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Missing model configuration: MATRIX_MODEL_PATH', result.stderr)
        self.assertIn('configs/paths.env.example', result.stderr)

    def test_cuda_stub_detection_uses_existing_paths_only(self):
        with tempfile.TemporaryDirectory(prefix='quantwm_release_cuda_') as temporary:
            home = Path(temporary)
            stub = home / 'lib64/stubs'
            stub.mkdir(parents=True)
            env = dict(CUDA_HOME=str(home), PATH='', LIBRARY_PATH=str(stub) + os.pathsep + '/custom/lib')
            configure_cuda_linker(env)
            values = env['LIBRARY_PATH'].split(os.pathsep)
            self.assertEqual(values[0], str(stub))
            self.assertEqual(len(values), len(set(values)))
            self.assertIn('/custom/lib', values)
            self.assertNotIn(str(home / 'targets/x86_64-linux/lib/stubs'), values)


class HistoricalStatisticsTests(unittest.TestCase):
    def test_full_query_energy_and_uniform_sample_are_separate(self):
        q = torch.arange(2 * 17 * 3 * 8, dtype=torch.float32).reshape(2, 17, 3, 8) / 100
        tracker = HistoricalQueryStatsTracker(layer_count=1, query_samples=16)
        tracker.observe(q)
        energy = q.square().sum(dim=(0, 1))
        torch.testing.assert_close(tracker.pending_sum[0], energy)
        self.assertEqual(tracker.pending_count[0], 34)
        self.assertEqual(tracker.get_queries(0).shape, (2, 3, 16, 8))
        indices = torch.linspace(0, 16, 16).round().long()
        torch.testing.assert_close(tracker.get_queries(0), q.index_select(1, indices).permute(0, 2, 1, 3))

    def test_uncommitted_queries_do_not_change_historical_sensitivity(self):
        q = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4) + 1
        tracker = HistoricalQueryStatsTracker(layer_count=1)
        tracker.observe(q)
        torch.testing.assert_close(tracker.get_weight(0, heads=2, dim=4, device='cpu'), torch.ones(2, 4))
        tracker.defer(tracker.drain_pending())
        self.assertNotIn(0, tracker.history_sum)
        tracker.commit_deferred()
        expected = q.square().sum((0, 1)) / 2
        expected = expected / expected.mean(-1, keepdim=True)
        torch.testing.assert_close(tracker.get_weight(0, heads=2, dim=4, device='cpu'), expected)


if __name__ == '__main__':
    unittest.main()
