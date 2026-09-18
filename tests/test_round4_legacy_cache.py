import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_offline_baseline import PROJECT_ROOT
import experiment_round3 as legacy


class LegacyCacheTests(unittest.TestCase):
    def test_adapter_change_cannot_return_old_round3_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ws=root/'workspace';ws.mkdir()
            names=('experiment_round3.py','api_client.py','zhipu.py')
            for name in names:(ws/name).write_text('first',encoding='utf-8')
            output=root/'results';output.mkdir();path=output/'dev-000-score-r0.json'
            row={'source_hash':'same-source','variant_config':legacy.VARIANTS['score'],'final_score':123}
            path.write_text(json.dumps(row),encoding='utf-8')
            with patch.object(legacy,'ROOT',root),patch.object(legacy,'code_hash',return_value='same-source'):
                with self.assertRaises(RuntimeError):legacy.run_one('dev-000','score',output=output)
                row['runner_fingerprint']=legacy.digest({'files':{name:hashlib.sha256((ws/name).read_bytes()).hexdigest() for name in names},
                                                        'model':{'model':'deterministic-first-choice'}})
                path.write_text(json.dumps(row),encoding='utf-8')
                self.assertEqual(legacy.run_one('dev-000','score',output=output)['final_score'],123)
                (ws/'zhipu.py').write_text('changed-model',encoding='utf-8')
                with self.assertRaises(RuntimeError):legacy.run_one('dev-000','score',output=output)


if __name__=='__main__':unittest.main()
