"""A second process must not launch a duplicate round9 batch."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

WORKSPACE = Path(__file__).resolve().parents[1] / 'workspace'
sys.path.insert(0, str(WORKSPACE))
from round9_lock import ExclusiveBatch


class LockTests(unittest.TestCase):
    def test_second_process_is_rejected_before_any_work(self):
        code = '''import sys
sys.path.insert(0, sys.argv[1])
from round9_lock import ExclusiveBatch
try:
    with ExclusiveBatch(sys.argv[2]):
        sys.exit(0)
except RuntimeError:
    sys.exit(23)
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'execution.lock'
            with ExclusiveBatch(path):
                child = subprocess.run([sys.executable, '-c', code, str(WORKSPACE), str(path)],
                                       capture_output=True, text=True, timeout=15)
            self.assertEqual(child.returncode, 23, child.stderr)
            child = subprocess.run([sys.executable, '-c', code, str(WORKSPACE), str(path)],
                                   capture_output=True, text=True, timeout=15)
            self.assertEqual(child.returncode, 0, child.stderr)

    def test_exception_releases_lock_without_deleting_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'execution.lock'
            with self.assertRaises(ValueError):
                with ExclusiveBatch(path):
                    raise ValueError('intentional unit-test exception')
            self.assertTrue(path.exists())
            with ExclusiveBatch(path):
                self.assertTrue(path.exists())


if __name__ == '__main__':
    unittest.main()
