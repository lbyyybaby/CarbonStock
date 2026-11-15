"""Check shot-level separation and the real loader's model inputs."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dataset.gedi_context import GediContext, masks_from_shots
from dataset.sentinel_gedi_canopy import GediSentinelDataset, PatchPair


class GediContextTest(unittest.TestCase):
    def test_nested_shot_roles_without_float_precision_loss(self):
        ids = np.arange(10**17, 10**17 + 1000, dtype=np.int64)
        shots = np.repeat(ids, 6).reshape(100, 60)
        valid = np.ones(shots.shape, dtype=bool)
        previous = np.zeros_like(valid)
        for fraction in (0, .1, .2, .5, .9):
            context, query = masks_from_shots(shots, valid, GediContext(True, fraction, 42))
            self.assertFalse((context & query).any())
            self.assertTrue((context | query).all())
            self.assertTrue((~previous | context).all())
            self.assertTrue(np.all(context.reshape(-1, 6) == context.reshape(-1, 6)[:, :1]))
            self.assertTrue(set(ids[context.reshape(-1, 6)[:, 0]]).isdisjoint(ids[query.reshape(-1, 6)[:, 0]]))
            # A crop/another patch must assign identical roles to the same IDs.
            a, b = masks_from_shots(shots[20:40], valid[20:40], GediContext(True, fraction, 42))
            np.testing.assert_array_equal(a, context[20:40])
            np.testing.assert_array_equal(b, query[20:40])
            previous = context
        for fraction in (-.1, 1, np.nan):
            with self.assertRaises(ValueError):
                GediContext(True, fraction)
        with self.assertRaises(ValueError):
            masks_from_shots(shots.astype(float), valid, GediContext(True))

    def test_loader_hides_labels_and_preserves_disabled_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ('input', 'gedi', 'input_valid_mask', 'shot_number'):
                (root / folder).mkdir()
            name = 'patch.npy'
            inputs = np.full((14, 10, 10), 2, dtype=np.float32)
            target = np.arange(100, dtype=np.float32).reshape(10, 10)
            shots = np.arange(10**17, 10**17 + 100, dtype=np.int64).reshape(10, 10)
            shots[0, 0] = 0
            valid = np.ones((10, 10), dtype=bool)
            valid[0, 1] = False
            for folder, values in [('input', inputs), ('gedi', target), ('input_valid_mask', valid), ('shot_number', shots)]:
                np.save(root / folder / name, values)
            pair = PatchPair('period', name, root/'input'/name, root/'gedi'/name,
                             root/'input_valid_mask'/name, root/'metadata.json', (0, 0, 100, 100), 99)
            mean, std = np.ones(14, dtype=np.float32), np.ones(14, dtype=np.float32)
            config = GediContext(True, .5)
            sample = GediSentinelDataset([pair], mean, std, max_agbd=80, gedi_context=config, target_scale=100)[0]
            context, query = masks_from_shots(shots, valid & (target <= 80), config)
            self.assertEqual(sample['image'].shape, (16, 10, 10))
            np.testing.assert_array_equal(sample['valid_mask'], query)
            np.testing.assert_array_equal(sample['image'][15], context.astype(np.float32))
            np.testing.assert_array_equal(sample['image'][14], np.where(context, target/100, 0))
            self.assertTrue(np.isnan(sample['mask'].numpy()[~query]).all())
            self.assertFalse(sample['image'][14].numpy()[query].any())
            # Perturbing query values cannot change either GEDI input channel.
            changed = target.copy(); changed[query] += .1
            np.save(root/'gedi'/name, changed)
            second = GediSentinelDataset([pair], mean, std, max_agbd=80, gedi_context=config, target_scale=100)[0]
            np.testing.assert_array_equal(sample['image'], second['image'])
            (root/'shot_number'/name).unlink()
            baseline = GediSentinelDataset([pair], mean, std)[0]
            self.assertEqual(baseline['image'].shape[0], 14)
            np.testing.assert_array_equal(baseline['valid_mask'], valid)


if __name__ == '__main__':
    unittest.main()
