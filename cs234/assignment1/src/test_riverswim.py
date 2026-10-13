import unittest

from riverswim import RiverCurrent, RiverSwimEnvr, RiverSwimResult

VAL_FUNC_LEFT = 30.328
VAL_FUNC_RIGHT = 36.859


class TestRiverSwim(unittest.TestCase):
    def setUp(self):
        self.seed = 1234
        self.current = RiverCurrent.WEAK
        self.gamma = 0.99
        self.tolerance = 1e-3
        self.env = RiverSwimEnvr(self.current, self.seed)

    def test_policy_iteration_rejects_invalid_gamma(self):
        with self.assertRaises(AssertionError):
            self.env.policy_iteration(gamma=1.5, tol=self.tolerance)

    def test_value_iteration_rejects_invalid_gamma(self):
        with self.assertRaises(AssertionError):
            self.env.value_iteration(gamma=1.5, tol=self.tolerance)

    def test_policy_iteration(self):
        result = self.env.policy_iteration(self.gamma, tol=self.tolerance)
        self.assertIsInstance(result, RiverSwimResult)
        self.assertAlmostEqual(result.val_func[0], VAL_FUNC_LEFT, places=3)
        self.assertAlmostEqual(result.val_func[-1], VAL_FUNC_RIGHT, places=3)

    def test_value_iteration(self):
        result = self.env.value_iteration(self.gamma, tol=self.tolerance)
        self.assertIsInstance(result, RiverSwimResult)
        self.assertAlmostEqual(result.val_func[0], VAL_FUNC_LEFT, places=3)
        self.assertAlmostEqual(result.val_func[-1], VAL_FUNC_RIGHT, places=3)


if __name__ == '__main__':
    unittest.main()
