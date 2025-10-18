import unittest

from distill.lego_response_distill import _register_teardown_hook, _invoke_teardown_for_test


class TeardownHookTests(unittest.TestCase):
    def setUp(self):
        _register_teardown_hook(None)

    def tearDown(self):
        _register_teardown_hook(None)

    def test_teardown_helper_invokes_registered_hook(self):
        calls = []

        def _hook(reason: str, exit_code: int) -> None:
            calls.append((reason, exit_code))

        _register_teardown_hook(_hook)
        _invoke_teardown_for_test("UNIT", 99)

        self.assertEqual(calls, [("UNIT", 99)])

    def test_teardown_helper_uses_default_arguments(self):
        captured = []

        def _hook(reason: str, exit_code: int) -> None:
            captured.append((reason, exit_code))

        _register_teardown_hook(_hook)
        _invoke_teardown_for_test()

        self.assertEqual(captured, [("TEST_SIGINT", 130)])


if __name__ == "__main__":
    unittest.main()
