"""Invariantes fisicos, causales y del protocolo; unittest sin dependencia extra."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch
import ris_vlc as rv
from entrenar import validate
from prediccion.data import episode, prepare, splits, Windows
from prediccion.models import Controller
from prediccion.physics import angles, project, segment_blocked, link
from prediccion.experiment import decide, setup


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).resolve().parents[1]/'config_prediccion.json').read_text())
        self.cfg.update(train_episodes=2, val_episodes=1, test_episodes=1, steps=20)
        self.esc = rv.Escenario(n_espejos=2)

    def test_config_rejects_noncausal_or_impossible_parameters(self):
        for changes in ({'history': 1}, {'dt': 0}, {'position_noise_m': -1},
                        {'angle_min_deg': [1, 1]}, {'angle_max_deg': [float('nan'), 80]}):
            cfg = dict(self.cfg, **changes)
            with self.assertRaises(ValueError):
                validate(cfg)

    def test_projection_respects_range_and_speed_after_many_commands(self):
        previous = np.zeros(8)
        rng = np.random.default_rng(5)
        for _ in range(100):
            theta = project(rng.normal(0, 10, 8), previous, self.cfg)
            self.assertTrue(np.all(abs(theta-previous) <= np.radians(120)*0.05 + 1e-14))
            self.assertTrue(np.all(abs(theta) <= np.radians(80)))
            previous = theta

    def test_cylinder_entry_below_top_and_vertical_segments(self):
        # Closest horizontal point is above the cylinder, but the segment clips
        # the rim below its top. A single closest-point test misses this case.
        self.assertTrue(segment_blocked([-2, 0, 3], [2, 0, 1], [0, 0, 0], 1, 1.8))
        self.assertTrue(segment_blocked([0, 0, 3], [0, 0, 0.8], [0, 0, 0], 0.2, 1.8))
        self.assertFalse(segment_blocked([-2, 0, 3], [2, 0, 3], [0, 0, 0], 1, 1.8))
        self.assertFalse(segment_blocked([0, 0, -3], [0, 0, -1], [0, 0, 0], 1, 1.8))
        self.assertTrue(segment_blocked([0, 0, 1], [0, 0, 1], [0, 0, 0], 1, 1.8))

    def test_reflected_path_blocked_and_unblocked_channel_parity(self):
        esc = self.esc
        pos, normal = np.array([1., 2.5, 0.85]), [0, 0, 1]
        theta = angles(esc, pos)
        rate, h, _ = link(esc, pos, normal, [10, 10, 0], theta, self.cfg)
        expected_h = rv.ganancia_irs(esc, pos, normal, rv.normales_optimas(esc, pos))
        self.assertAlmostEqual(h, expected_h, places=16)
        self.assertAlmostEqual(rate, float(rv.tasa(esc, expected_h + rv.ganancia_los(esc, pos, normal))), places=12)
        cfg = dict(self.cfg, blocker_radius_m=10, blocker_height_m=4)
        rate, h, blocked = link(esc, pos, normal, [2, 2, 0], theta, cfg)
        self.assertEqual((rate, h, blocked), (0, 0, True))

    def test_roundtrip_normals_and_angle_interleaving(self):
        pos = [1, 3, 0.85]
        theta = angles(self.esc, pos).reshape(-1, 2)
        np.testing.assert_allclose(rv.normal_desde_angulos(theta[:, 0], theta[:, 1]),
                                   rv.normales_optimas(self.esc, pos), atol=1e-14)

    def test_noise_sweep_keeps_identical_latent_world(self):
        clean = episode(dict(self.cfg, position_noise_m=0.0), self.esc, 12)
        noisy = episode(dict(self.cfg, position_noise_m=0.10), self.esc, 12)
        for key in ('position', 'normal', 'blocker', 'observed_normal'):
            np.testing.assert_array_equal(clean[key], noisy[key])
        np.testing.assert_array_equal(clean['observed'], clean['position'])
        self.assertFalse(np.array_equal(clean['observed'], noisy['observed']))

    def test_seeds_reproducible_disjoint_and_windows_stay_in_episode(self):
        data = splits(self.cfg, self.esc)
        seeds = [ep['seed'] for eps in data.values() for ep in eps]
        self.assertEqual(len(set(seeds)), len(seeds))
        first = data['train'][0]
        again = episode(self.cfg, self.esc, first['seed'])
        np.testing.assert_array_equal(first['observed'], again['observed'])
        windows = Windows(data['train'], self.cfg)
        maps, coords, base, target = windows[0]
        np.testing.assert_array_equal(target, first['target'][self.cfg['history']])
        self.assertEqual(len(maps), self.cfg['history'])
        self.assertTrue(all(t >= self.cfg['history']-1 and t+1 < self.cfg['steps']
                            for _, t in windows.index))

    def test_no_controller_reads_future_truth_or_measurements(self):
        setup(self.cfg)
        ep = prepare(episode(self.cfg, self.esc, 10), self.esc, self.cfg)
        t = self.cfg['history']-1
        altered = copy.deepcopy(ep)
        # Future measurements and ALL latent truth can change without affecting
        # the decision at t. Future labels are legal only in training/evaluation.
        for key in ['maps', 'coords', 'base', 'observed', 'observed_normal', 'blocker']:
            altered[key][t+1:] = 100
        altered['position'][:] = 100
        altered['normal'][:] = 100
        altered['target'][:] = 100
        models = {kind: Controller(8, kind).eval() for kind in ('cnn_gru', 'coord_gru', 'cnn')}
        # Make the neural heads nonzero so this tests input dependence.
        for model in models.values():
            torch.nn.init.normal_(model.head.weight, std=0.1)
        previous = np.zeros(8)
        for method in ['reactive', 'constant_velocity', 'geometric_j', *models]:
            np.testing.assert_array_equal(decide(method, ep, t, previous, self.esc, self.cfg, models),
                                          decide(method, altered, t, previous, self.esc, self.cfg, models))

    def test_model_gradients_are_finite_and_optimization_reduces_loss(self):
        setup(self.cfg)
        ep = prepare(episode(self.cfg, self.esc, 10), self.esc, self.cfg)
        window = Windows([ep], self.cfg)[0]
        inputs = [x.unsqueeze(0) for x in window[:3]]
        target = window[3].unsqueeze(0)
        for kind in ('cnn_gru', 'coord_gru', 'cnn'):
            model = Controller(8, kind)
            opt = torch.optim.Adam(model.parameters(), lr=0.002)
            initial = ((model(*inputs)-target)**2).mean().item()
            for _ in range(30):
                loss = ((model(*inputs)-target)/0.01).square().mean()
                opt.zero_grad()
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))
                opt.step()
            self.assertLess(((model(*inputs)-target)**2).mean().item(), initial)


if __name__ == '__main__':
    unittest.main()
