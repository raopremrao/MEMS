import unittest

import numpy as np

from microgrid_eval import (
    CoordinatedController,
    MicrogridConfig,
    MicrogridEnv,
    REPORT_REFERENCE_RESULTS,
    RuleBasedController,
    UncoordinatedController,
    evaluate_strategies,
)


class MicrogridEvaluationTests(unittest.TestCase):
    def test_profiles_are_reproducible_and_have_expected_length(self):
        first = MicrogridEnv().generate_daily_profiles()
        second = MicrogridEnv().generate_daily_profiles()
        self.assertEqual(len(first[0]), 96)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)

    def test_strategies_produce_finite_metrics(self):
        results = evaluate_strategies()
        self.assertEqual(len(results), 3)
        numeric = results.select_dtypes(include=[np.number]).to_numpy()
        self.assertTrue(np.isfinite(numeric).all())
        self.assertTrue((results["Daily operating cost ($)"] >= 0).all())
        self.assertTrue((results["Peak grid power (kW)"] >= 0).all())

    def test_soc_stays_within_limits_after_enforcement(self):
        env = MicrogridEnv(MicrogridConfig())
        for controller in (RuleBasedController(), UncoordinatedController(), CoordinatedController()):
            result = env.run(controller)
            self.assertGreaterEqual(result["Final state of charge"], env.config.soc_min)
            self.assertLessEqual(result["Final state of charge"], env.config.soc_max)

    def test_report_values_are_reference_data(self):
        self.assertEqual(REPORT_REFERENCE_RESULTS["Daily operating cost ($)"].tolist(),
                         [48.60, 43.10, 39.65])
        self.assertEqual(REPORT_REFERENCE_RESULTS["Battery limit violations"].tolist(), [0, 4, 0])


if __name__ == "__main__":
    unittest.main()
