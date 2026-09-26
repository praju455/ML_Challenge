from __future__ import annotations
import csv, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from src.features import FEATURE_COLUMNS
from src.predict import predict_file

class PredictTests(unittest.TestCase):
 def test_writes_empty_rows_and_thresholded_matches(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d); src=r/'s1.tsv'; feat=r/'f.tsv'; out=r/'o.tsv'; src.write_text('entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tx\tx\tx\nS1-2\tx\tx\tx\n')
   with feat.open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=FEATURE_COLUMNS, delimiter='\t'); w.writeheader(); w.writerow(dict.fromkeys(FEATURE_COLUMNS, ''))
    f.seek(0) # replace with a compact valid row below
   row={c:'0' for c in FEATURE_COLUMNS}; row.update(source1_entity_id='S1-1',candidate_entity_id='S2-1',is_match='')
   with feat.open('w', newline='') as f: w=csv.DictWriter(f,fieldnames=FEATURE_COLUMNS,delimiter='\t');w.writeheader();w.writerow(row)
   class M:
    def predict_proba(self,x): return [[.1,.9]]
   predict_file(feat,r/'m',src,out,model_loader=lambda _:{'feature_columns': FEATURE_COLUMNS[3:], 'threshold':.8, 'calibrator':M()})
   with out.open() as f: rows=list(csv.DictReader(f,delimiter='\t'))
   self.assertEqual(rows,[{'source1_entity_id':'S1-1','matched_entity_ids':'S2-1'},{'source1_entity_id':'S1-2','matched_entity_ids':''}])
