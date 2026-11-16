"""Invariantes del ensayo de robustez, sin entrenamiento ni checkpoints."""
import json
from pathlib import Path
import unittest

import numpy as np

import ris_vlc as rv
from prediccion.data import episode, prepare
from prediccion.stress import SCENARIOS, make_scenario, scenario_metadata


class StressScenarioTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).resolve().parents[1]/'config_prediccion.json').read_text())
        self.cfg.update(steps=60, history=8)
        self.esc = rv.Escenario(n_espejos=2)
        self.seed = 20261179
        self.event = 30
        self.original = episode(self.cfg, self.esc, self.seed)

    def make(self, scenario):
        return make_scenario(self.cfg, self.esc, self.seed, scenario, self.event)

    def test_normal_exactly_matches_existing_generator_and_all_are_reproducible(self):
        for scenario in SCENARIOS:
            first, second = self.make(scenario), self.make(scenario)
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
                if scenario == 'normal':
                    np.testing.assert_array_equal(first[key], self.original[key])

    def test_paired_prefix_blocker_and_sensor_noise(self):
        for scenario in SCENARIOS:
            ep = self.make(scenario)
            np.testing.assert_array_equal(ep['blocker'], self.original['blocker'])
            if scenario != 'stationary':
                for key in ('position', 'normal', 'blocker', 'observed', 'observed_normal'):
                    np.testing.assert_array_equal(ep[key][:self.event], self.original[key][:self.event])
            np.testing.assert_allclose(
                ep['observed'][:, :2] - ep['position'][:, :2],
                self.original['observed'][:, :2] - self.original['position'][:, :2], atol=1e-14)
            np.testing.assert_array_equal(ep['observed'][:, 2], ep['position'][:, 2])
            # The same angular sensing error survives the rotation during fall.
            np.testing.assert_allclose(np.sum(ep['normal']*ep['observed_normal'], axis=1),
                                       np.sum(self.original['normal']*self.original['observed_normal'], axis=1),
                                       atol=1e-14)

    def test_stopping_and_stationary_positions(self):
        stop, stationary = self.make('sudden_stop'), self.make('stationary')
        np.testing.assert_array_equal(stop['position'][self.event-1:],
                                      np.repeat(self.original['position'][self.event-1:self.event],
                                                self.cfg['steps']-self.event+1, axis=0))
        np.testing.assert_array_equal(stationary['position'],
                                      np.repeat(self.original['position'][:1], self.cfg['steps'], axis=0))
        self.assertGreater(np.linalg.norm(np.diff(self.original['position'][:self.event], axis=0)), 0)

    def test_erratic_is_bounded_and_no_teleportation(self):
        for seed in range(self.seed, self.seed+16):
            ep = make_scenario(self.cfg, self.esc, seed, 'erratic', self.event)
            position = ep['position']
            self.assertTrue(np.all(position[:, :2] >= 0.35-1e-12))
            self.assertTrue(np.all(position[:, :2] <= self.esc.sala[:2]-0.35+1e-12))
            steps = np.diff(position, axis=0)
            self.assertTrue(np.all(np.linalg.norm(steps, axis=1) <= self.cfg['max_speed_m_s']*self.cfg['dt']+1e-12))
            # First changed displacement sharply departs from the preceding one.
            self.assertLessEqual(np.dot(steps[self.event-2], steps[self.event-1]), 1e-12)

    def test_fall_is_continuous_vertical_transition_and_then_fixed(self):
        ep = self.make('fall_proxy')
        meta = scenario_metadata(self.cfg, self.esc, self.event)
        start, end = self.event-1, meta['fall_end_index']
        np.testing.assert_array_equal(ep['position'][start:, :2],
                                      np.repeat(ep['position'][start:start+1, :2], self.cfg['steps']-start, axis=0))
        self.assertTrue(np.all(np.diff(ep['position'][start:, 2]) <= 0))
        self.assertLess(abs(ep['position'][self.event, 2]-ep['position'][start, 2]), 0.03)
        np.testing.assert_allclose(ep['position'][end:, 2], 0.25)
        np.testing.assert_allclose(ep['normal'][end:, 2], 0.0)
        np.testing.assert_allclose(np.linalg.norm(ep['normal'], axis=1), 1.0)
        np.testing.assert_allclose(np.linalg.norm(ep['observed_normal'], axis=1), 1.0)
        np.testing.assert_array_equal(ep['normal'][end:],
                                      np.repeat(ep['normal'][end:end+1], self.cfg['steps']-end, axis=0))

    def test_preparation_uses_current_height_without_changing_map_schema(self):
        fall = self.make('fall_proxy')
        data = prepare(fall, self.esc, self.cfg)
        self.assertEqual(data['maps'].shape, (self.cfg['steps'], 8, self.cfg['grid'], self.cfg['grid']))
        np.testing.assert_allclose(data['coords'][:, 2], fall['position'][:, 2]/self.esc.sala[2])
        self.assertFalse(np.array_equal(data['base'][self.event-1], data['base'][-1]))
        flattened_height = {key: value.copy() if isinstance(value, np.ndarray) else value
                            for key, value in fall.items()}
        flattened_height['observed'][:, 2] = self.esc.altura_dispositivo
        flat_data = prepare(flattened_height, self.esc, self.cfg)
        np.testing.assert_array_equal(data['maps'], flat_data['maps'])
        self.assertFalse(np.array_equal(data['base'][-1], flat_data['base'][-1]))

    def test_invalid_scenario_event_and_incomplete_fall_fail(self):
        with self.assertRaises(ValueError):
            self.make('unknown')
        for index in (0, 7, self.cfg['steps'], 20.5, True):
            with self.assertRaises(ValueError):
                make_scenario(self.cfg, self.esc, self.seed, 'sudden_stop', index)
        with self.assertRaises(ValueError):
            make_scenario(self.cfg, self.esc, self.seed, 'fall_proxy', self.cfg['steps']-2)


if __name__ == '__main__':
    unittest.main()
