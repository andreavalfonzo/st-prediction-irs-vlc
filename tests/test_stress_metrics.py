"""Ventanas, unidades estadisticas y auditorias de la evaluacion congelada."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluar_estres import (
    in_window, paired_differences, summarize,
    verify_causal_prefix, verify_original_normal,
)


class StressMetricsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(dt=0.05, history=8, steps=120, test_episodes=2,
                        seed=20261103, r_min=5.0)
        self.metadata = dict(event_index=60)

    def test_window_boundaries_use_application_sample(self):
        for index, expected in (
                (59, {'all', 'before'}),
                (60, {'all', 'onset', 'transition', 'after'}),
                (69, {'all', 'transition', 'after'}),
                (70, {'all', 'after', 'settled'})):
            actual = {window for window in ('all', 'before', 'onset', 'transition', 'after', 'settled')
                      if in_window(index, window, self.cfg, self.metadata)}
            self.assertEqual(actual, expected)
        # Settled requires the longest of transition duration and history.
        long_history = dict(self.cfg, history=12)
        self.assertFalse(in_window(71, 'settled', long_history, self.metadata))
        self.assertTrue(in_window(72, 'settled', long_history, self.metadata))

    def test_summaries_bootstrap_episodes_and_combine_angular_squared_error(self):
        rows = []
        # The two post-event episode means are 4 and 8. Pooling individual
        # interval rates for bootstrap would instead yield a different CI.
        for episode_id, rates, errors in ((0, (0.0, 8.0), (3.0, 4.0)),
                                          (1, (4.0, 12.0), (0.0, 12.0))):
            for application, rate, error in ((59, 100.0, 20.0),
                                               (60, rates[0], errors[0]),
                                               (61, rates[1], errors[1])):
                rows.append(dict(scenario='sudden_stop', method='cnn_gru',
                                 episode=episode_id, seed=101+episode_id,
                                 t=application-1, rate=rate, angle_rmse_deg=error,
                                 blocked=int(rate == 0), h_irs=rate*1e-8))
        summaries, per_episode = summarize(rows, self.cfg, self.metadata)
        lookup = {row['window']: row for row in summaries}
        onset, after = lookup['onset'], lookup['after']
        self.assertEqual(onset['samples'], 2)
        self.assertEqual(onset['mean_rate'], 2.0)
        self.assertEqual(after['episodes'], 2)
        self.assertEqual(after['samples'], 4)
        self.assertEqual(after['mean_rate'], 6.0)
        self.assertEqual(after['rate_ci_low'], 4.0)
        self.assertEqual(after['rate_ci_high'], 8.0)
        self.assertEqual(after['outage_fraction'], 0.5)
        self.assertEqual(after['blocked_fraction'], 0.25)
        self.assertEqual(after['angle_rmse_deg'], 6.5)
        self.assertEqual(after['peak_angle_rmse_deg'], 12.0)
        episodes = {row['episode']: row for row in per_episode if row['window'] == 'after'}
        self.assertEqual(episodes[0]['angle_mse_deg2'], 12.5)
        self.assertEqual(episodes[1]['angle_mse_deg2'], 72.0)
        self.assertEqual(episodes[0]['samples'], 2)
        self.assertNotIn('settled', lookup)

    def paired_fixture(self):
        rows = []
        for scenario, method, rates, outages, errors in (
                ('normal', 'reactive', (2., 10.), (.5, 0.), (9., 25.)),
                ('normal', 'cnn_gru', (4., 12.), (0., 0.), (1., 9.)),
                ('sudden_stop', 'reactive', (0., 8.), (1., .5), (16., 36.)),
                ('sudden_stop', 'cnn_gru', (1., 9.), (.5, .5), (4., 16.))):
            for i, seed in enumerate((101, 102)):
                # Episode IDs intentionally disagree across groups; only seed
                # identifies a matched world, and input ordering is irrelevant.
                rows.append(dict(scenario=scenario, method=method, window='after',
                                 episode=i if scenario == 'normal' else 1-i,
                                 seed=seed, samples=10, mean_rate=rates[i],
                                 outage_fraction=outages[i], angle_mse_deg2=errors[i]))
        return rows[::-1]

    def test_paired_differences_match_seeds_before_bootstrap(self):
        rows = paired_differences(self.paired_fixture(), self.cfg)
        lookup = {(row['scenario'], row['method'], row['contrast']): row for row in rows}
        versus_reactive = lookup[('sudden_stop', 'cnn_gru', 'versus_reactive')]
        self.assertEqual(versus_reactive['episodes'], 2)
        self.assertEqual(versus_reactive['delta_mean_rate'], 1.0)
        # Each paired difference is +1 although the individual episode rates
        # differ by 8. Independent bootstraps would not produce this zero width.
        self.assertEqual(versus_reactive['delta_rate_ci_low'], 1.0)
        self.assertEqual(versus_reactive['delta_rate_ci_high'], 1.0)
        self.assertEqual(versus_reactive['delta_outage_pp'], -25.0)
        self.assertAlmostEqual(versus_reactive['delta_angle_rmse_deg'], np.sqrt(10)-np.sqrt(26))
        versus_normal = lookup[('sudden_stop', 'cnn_gru', 'versus_normal')]
        self.assertEqual(versus_normal['delta_mean_rate'], -3.0)
        self.assertEqual(versus_normal['delta_rate_ci_low'], -3.0)
        self.assertEqual(versus_normal['delta_rate_ci_high'], -3.0)
        self.assertEqual(versus_normal['delta_outage_pp'], 50.0)
        self.assertAlmostEqual(versus_normal['delta_angle_rmse_deg'], np.sqrt(10)-np.sqrt(5))

    def test_paired_differences_reject_unmatched_seeds(self):
        rows = self.paired_fixture()
        rows[0]['seed'] = 103
        with self.assertRaisesRegex(ValueError, 'mismas semillas'):
            paired_differences(rows, self.cfg)

    def write_angle_fixture(self, output):
        length = self.cfg['steps']-self.cfg['history']
        array = np.arange(self.cfg['test_episodes']*length*4, dtype='float32').reshape(-1, 4)
        for scenario in ('normal', 'sudden_stop', 'erratic', 'fall_proxy'):
            folder = output/scenario
            folder.mkdir()
            np.savez(folder/'applied_angles.npz', reactive=array, cnn_gru=array+0.5)
        return array

    def test_causal_prefix_includes_first_affected_application_for_each_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            array = self.write_angle_fixture(output)
            self.assertTrue(verify_causal_prefix(output, self.cfg, self.metadata))
            length = self.cfg['steps']-self.cfg['history']
            event_column = self.metadata['event_index']-self.cfg['history']
            candidate = (array+0.5).reshape(self.cfg['test_episodes'], length, 4).copy()
            # A command at event+1 may legitimately see the changed observation.
            candidate[1, event_column+1, 0] += 5
            path = output/'erratic'/'applied_angles.npz'
            np.savez(path, reactive=array, cnn_gru=candidate.reshape(-1, 4))
            self.assertTrue(verify_causal_prefix(output, self.cfg, self.metadata))
            # The preceding command was chosen before the first perturbation:
            # a difference here proves a leaked event or unpaired initialization.
            candidate[1, event_column, 0] += 5
            np.savez(path, reactive=array, cnn_gru=candidate.reshape(-1, 4))
            with self.assertRaisesRegex(AssertionError, 'erratic/cnn_gru'):
                verify_causal_prefix(output, self.cfg, self.metadata)

    def test_original_audit_ignores_latency_and_order_but_detects_channel_changes(self):
        def write(path, rows):
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        with tempfile.TemporaryDirectory() as folder:
            checkpoint, output = Path(folder)/'checkpoint', Path(folder)/'output'
            checkpoint.mkdir()
            (output/'normal').mkdir(parents=True)
            self.assertEqual(verify_original_normal(checkpoint, output), {'available': False})
            original = [dict(method='cnn_gru', episode=i, seed=101+i, t=59,
                             rate=8.+i, h_irs=1e-7, blocked=0, movement_deg=3.,
                             angle_rmse_deg=.5, j=7., decision_ms=1.) for i in range(2)]
            updated = copy.deepcopy(original[::-1])
            for row in updated:
                row['decision_ms'] = 100.0
            write(checkpoint/'traces.csv', original)
            write(output/'normal'/'traces.csv', updated)
            audit = verify_original_normal(checkpoint, output)
            self.assertTrue(audit['reproduced'])
            self.assertTrue(all(value == 0 for value in audit['max_absolute_differences'].values()))
            updated[0]['rate'] += .5
            write(output/'normal'/'traces.csv', updated)
            with self.assertRaisesRegex(AssertionError, 'original: rate'):
                verify_original_normal(checkpoint, output)


if __name__ == '__main__':
    unittest.main()
