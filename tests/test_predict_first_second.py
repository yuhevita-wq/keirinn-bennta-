"""Independent first+conditional-second research probability checks."""
import json
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from predict_first_second import predict_first_second
from test_predict_first import fixture

FIRST=json.loads((ROOT/"models/first_win_2026sep_manifest.json").read_text(encoding="utf-8"))
SECOND=json.loads((ROOT/"models/second_conditional_v1.json").read_text(encoding="utf-8"))

class SecondModelTests(unittest.TestCase):
    def test_all_ordered_pairs_and_normalization(self):
        d=predict_first_second(fixture(),FIRST,SECOND)
        pairs=d["ordered_first_second_pairs"]
        self.assertEqual(len(pairs),7*6)
        self.assertTrue(all(x["first_car_no"]!=x["second_car_no"] for x in pairs))
        self.assertAlmostEqual(sum(x["joint_first_second_probability"] for x in pairs),1,places=9)
        self.assertAlmostEqual(sum(x["probability"] for x in d["second_marginal"]),1,places=9)
        self.assertEqual(len(d["first_marginal"]),7)
        self.assertEqual(len(d["second_marginal"]),7)

    def test_conditional_probability_sums_to_1_for_every_possible_winner(self):
        d=predict_first_second(fixture(),FIRST,SECOND)
        for i in range(1,8):
            conditional=sum(x["conditional_second_given_first"]
                            for x in d["ordered_first_second_pairs"] if x["first_car_no"]==i)
            self.assertAlmostEqual(conditional,1.0,places=9)

    def test_other_race_classes_not_allowed(self):
        c=fixture()
        c["race_type"]="Ａ級予選"
        with self.assertRaises(ValueError):
            predict_first_second(c,FIRST,SECOND)

    def test_no_backfill_from_known_actual_winner(self):
        c=fixture()
        c["result"]={"first":1,"second":2}
        with self.assertRaises(ValueError):
            predict_first_second(c,FIRST,SECOND)

    def test_frozen_four_parameter_model_is_identifiable(self):
        co=SECOND["coefficients"]
        self.assertEqual(set(co),{
            "gamma","directly_behind_winner","ahead_of_winner",
            "later_behind_winner","other_line"
        })
        self.assertAlmostEqual(co["gamma"],0.6407563020428341,places=10)
        self.assertEqual(SECOND["fit_sample_races"],14081)
        self.assertEqual(SECOND["fit_sample_second_candidate_rows"],83381)

if __name__=="__main__":
    unittest.main()
