import unittest
import subprocess

# ------------------------------------------------------------------------------
# Test file for verifying that the main driver script `testproject.py`
# runs correctly and produces expected key output messages.
#
# This test runs `testproject.py` as a separate subprocess (like from command line),
# captures its stdout and stderr, and asserts the presence of expected print statements.
#
# Benefits of this approach:
# - Avoids import-time execution issues with top-level code in `testproject.py`
# - Captures printed output as seen by a user running the script
# - Confirms the script completes without error (return code 0)
# ------------------------------------------------------------------------------

class TestTestProject(unittest.TestCase):
    def test_main_run(self):
        # Run testproject.py as a subprocess, capturing stdout and stderr
        result = subprocess.run(
            ['python3', 'testproject.py'],  # Run script with Python 3 interpreter
            stdout=subprocess.PIPE,         # Capture standard output
            stderr=subprocess.PIPE,         # Capture error output
            text=True                      # Return outputs as string instead of bytes
        )
        
        # Combine stdout and stderr for comprehensive output checking
        all_output = result.stdout + result.stderr

        # Assert that key printed messages from testproject.py appear in the output
        self.assertIn("Running Manual Strategy...", all_output)
        self.assertIn("Running Strategy Learner...", all_output)
        self.assertIn("Running Experiment 1: Comparing strategies...", all_output)
        self.assertIn("Running Experiment 2: Varying impact analysis...", all_output)
        self.assertIn("All tasks completed successfully!", all_output)

        # Assert that the subprocess exited successfully (no runtime errors)
        self.assertEqual(result.returncode, 0)

if __name__ == "__main__":
    unittest.main()

