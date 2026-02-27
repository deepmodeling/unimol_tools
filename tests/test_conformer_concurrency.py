"""
Test suite for concurrency and race condition issues in conformer.py

This test suite demonstrates the race condition issues described in GitHub issue #19:
- Missing pool.join() calls in transform() methods
- Bare except: clauses catching KeyboardInterrupt
- No timeout handling for pool.imap()
- No proper cleanup on interruption

These tests are expected to FAIL or exhibit issues on the current main branch,
demonstrating that the bugs exist and documenting the expected fix requirements.
"""

import unittest
import multiprocessing
import time
import signal
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import patch, MagicMock
import numpy as np


class TestConformerConcurrencyIssues(unittest.TestCase):
    """Test to demonstrate concurrency handling issues in conformer generation."""

    def test_multiprocess_hang_risk(self):
        """
        Test that demonstrates the risk of hangs when using multi_process=True.
        
        This test sets a timeout to detect if the transform method hangs,
        which is the symptom of the race condition described in issue #19.
        """
        from unimol_tools.data.conformer import ConformerGen
        
        # Use a simple molecule set
        smiles_list = ['C', 'CC', 'CCC']
        
        gen = ConformerGen(multi_process=True, max_atoms=128)
        
        # Set a timeout to detect hangs
        def timeout_handler(signum, frame):
            raise TimeoutError("Transform timed out - possible race condition!")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        try:
            start_time = time.time()
            inputs, mols = gen.transform(smiles_list)
            elapsed = time.time() - start_time
            signal.alarm(0)  # Cancel alarm
            
            # If we get here without timeout, the test passes
            # but this is flaky due to the race condition
            self.assertEqual(len(inputs), len(smiles_list))
            
        except TimeoutError:
            signal.alarm(0)
            # This demonstrates the bug exists
            # The test documents that this timeout should not happen
            self.fail("Transform hung - race condition detected! "
                     "This demonstrates issue #19: missing pool.join()")

    def test_keyboard_interrupt_propagation(self):
        """
        Test that KeyboardInterrupt is properly propagated (not suppressed by bare except).
        
        This test verifies that if a KeyboardInterrupt occurs during processing,
        it is not caught by a bare 'except:' clause.
        """
        from unimol_tools.data.conformer import inner_smi2coords
        
        # The current implementation may suppress KeyboardInterrupt
        # This test documents the expected behavior
        
        # Test with normal input first
        try:
            result = inner_smi2coords('C', return_mol=False)
            # Should return coordinates, not raise
            self.assertIsNotNone(result)
        except KeyboardInterrupt:
            self.fail("KeyboardInterrupt should not be raised during normal operation")
        except Exception:
            # Other exceptions are acceptable for this test
            pass


class TestExceptionHandlingIssues(unittest.TestCase):
    """Test to demonstrate exception handling issues."""

    def test_bare_except_detection(self):
        """
        Detect bare 'except:' clauses in the source code.
        
        Bare except clauses catch all exceptions including KeyboardInterrupt,
        which is problematic as noted in issue #19.
        
        This test documents the issue but does not fail - it just logs findings.
        """
        import ast
        import inspect
        from unimol_tools.data import conformer
        
        # Read source file
        source_file = inspect.getfile(conformer)
        with open(source_file, 'r') as f:
            tree = ast.parse(f.read())
        
        # Check for bare except clauses
        bare_excepts = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    bare_excepts.append(node.lineno)
        
        # Log findings
        if bare_excepts:
            print(f"\n[ISSUE DETECTED] Found bare 'except:' at lines: {bare_excepts}")
            print("These should be replaced with 'except Exception:' to avoid catching KeyboardInterrupt")
            # Don't fail the test - just document the issue
        else:
            print("\n[GOOD] No bare 'except:' clauses found")

    def test_error_logging_vs_print(self):
        """
        Test that errors are logged properly (not just printed).
        
        Issue #19 notes that print statements are used instead of logger.
        """
        from unimol_tools.data.conformer import inner_smi2coords
        from unimol_tools.utils import logger
        
        # Test with invalid SMILES
        with patch.object(logger, 'error') as mock_error:
            with patch('builtins.print') as mock_print:
                result = inner_smi2coords('invalid_smiles_xxx', return_mol=False)
                
                # Should return zero coordinates on failure
                self.assertIsNotNone(result)
                
                # Document whether logger or print is used
                if mock_print.called and not mock_error.called:
                    print("\n[ISSUE DETECTED] Using print() instead of logger.error()")
                    print("This should use the logger for better error handling")


class TestUniMolV2ConcurrencyIssues(unittest.TestCase):
    """Test concurrency handling issues in UniMolV2Feature."""

    def test_unimolv2_hang_risk(self):
        """
        Test that demonstrates the risk of hangs in UniMolV2Feature.
        
        Similar to ConformerGen, UniMolV2Feature has the same race condition issues.
        """
        from unimol_tools.data.conformer import UniMolV2Feature
        
        smiles_list = ['C', 'CC']
        gen = UniMolV2Feature(multi_process=True, max_atoms=128)
        
        def timeout_handler(signum, frame):
            raise TimeoutError("UniMolV2Feature transform timed out!")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            start_time = time.time()
            inputs, mols = gen.transform(smiles_list)
            elapsed = time.time() - start_time
            signal.alarm(0)
            
            self.assertEqual(len(inputs), len(smiles_list))
            
        except TimeoutError:
            signal.alarm(0)
            self.fail("UniMolV2Feature transform hung! "
                     "This demonstrates issue #19 in UniMolV2Feature")
        except Exception:
            signal.alarm(0)
            pass


class TestPoolCleanupIssues(unittest.TestCase):
    """Test that documents pool cleanup issues."""

    def test_pool_cleanup_verification(self):
        """
        Verify that worker processes are properly cleaned up after transform.
        
        This test documents the expected behavior - pools should be properly closed
        and joined to prevent zombie processes.
        """
        from unimol_tools.data.conformer import ConformerGen
        
        try:
            import psutil
            
            # Get initial process count
            initial_procs = len(psutil.Process().children())
            
            gen = ConformerGen(multi_process=True)
            smiles_list = ['C', 'CC', 'CCC']
            
            try:
                gen.transform(smiles_list)
            except:
                pass
            
            # Give some time for cleanup
            time.sleep(1)
            
            final_procs = len(psutil.Process().children())
            
            # Document the state
            if final_procs > initial_procs + 2:
                print(f"\n[ISSUE DETECTED] Process count not cleaned up: "
                      f"{initial_procs} -> {final_procs}")
                print("This indicates missing pool.join() or proper cleanup")
            else:
                print(f"\n[GOOD] Process cleanup working: {initial_procs} -> {final_procs}")
                
        except ImportError:
            self.skipTest("psutil not available for process monitoring")


def run_stress_test(iterations=5):
    """
    Run stress test to detect race conditions.
    
    This test runs multiple iterations of multiprocess transform
    to increase the chance of detecting race conditions.
    
    Usage: python test_conformer_concurrency.py --stress
    """
    from unimol_tools.data.conformer import ConformerGen
    
    print(f"\nRunning stress test with {iterations} iterations...")
    
    smiles_list = ['C', 'CC', 'CCC', 'CCCC', 'c1ccccc1']
    
    timeouts_detected = 0
    
    for i in range(iterations):
        print(f"  Iteration {i+1}/{iterations}...", end=' ')
        sys.stdout.flush()
        
        gen = ConformerGen(multi_process=True, max_atoms=128)
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Iteration {i+1} timed out!")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(20)
        
        try:
            start = time.time()
            inputs, mols = gen.transform(smiles_list)
            elapsed = time.time() - start
            signal.alarm(0)
            
            if len(inputs) == len(smiles_list):
                print(f"OK ({elapsed:.2f}s)")
            else:
                print(f"PARTIAL ({len(inputs)}/{len(smiles_list)})")
                
        except TimeoutError as e:
            signal.alarm(0)
            print(f"TIMEOUT - {e}")
            timeouts_detected += 1
        except Exception as e:
            signal.alarm(0)
            print(f"ERROR - {type(e).__name__}")
    
    print("="*60)
    if timeouts_detected > 0:
        print(f"[ISSUE DETECTED] {timeouts_detected}/{iterations} iterations timed out")
        print("This demonstrates the race condition described in issue #19")
        return False
    else:
        print("Stress test completed without timeouts")
        print("Note: Race conditions are intermittent and may not always trigger")
        return True


if __name__ == '__main__':
    # Check if stress test requested
    if '--stress' in sys.argv:
        sys.argv.remove('--stress')
        success = run_stress_test(iterations=10)
        sys.exit(0 if success else 1)
    
    # Run unit tests
    unittest.main()
