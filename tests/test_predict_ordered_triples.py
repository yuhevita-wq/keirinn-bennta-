"""Check full ordered triple probability with adopted 1st/2nd/3rd mathematical rules."""
import json
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from predict_ordered_triples import infer_triples,third_position,third_role
from test_predict_first import fixture

FIRST=json.loads((ROOT/"models/first_win_2026sep_manifest.json").read_text(encoding="utf-8"))
SECOND=json.loads((ROOT/"models/second_conditional_v1.json").read_text(encoding="utf-8"))
THIRD=json.loads((ROOT/"models/third_conditional_v1.json").read_text(encoding="utf-8"))

class TripleProbabilityTests(unittest.TestCase):
    def test_all_distinct_triples_preserve_two_place_model(self):
        v=infer_triples(fixture(),FIRST,SECOND,THIRD)
        t=v["ordered_first_second_third_triples"]
        self.assertEqual(len(t),210)
        self.assertEqual(v["all_ordered_triples_count"],210)
        keys={(x["first_car_no"],x["second_car_no"],x["third_car_no"]) for x in t}
        self.assertEqual(len(keys),210)
        self.assertTrue(all(len({i,j,k})==3 for i,j,k in keys))
        self.assertAlmostEqual(sum(x["probability"] for x in t),1.0,places=9)
        for name in ("first_marginal","second_marginal","third_marginal"):
            self.assertAlmostEqual(sum(x["probability"] for x in v[name]),1.0,places=9)
        for i in range(1,8):
            fi=next(x["probability"] for x in v["first_marginal"] if x["car_no"]==i)
            from_triples=sum(x["probability"] for x in t if x["first_car_no"]==i)
            self.assertAlmostEqual(fi,from_triples,places=9)
        self.assertEqual(v,infer_triples(fixture(),FIRST,SECOND,THIRD))

    def test_first_second_third_line_positions(self):
        lines=[[1,2,3],[4,5],[6,7]]
        self.assertEqual(third_position(3,1,2,lines),"behind_both")
        self.assertEqual(third_position(1,2,3,lines),"ahead_of_both")
        self.assertEqual(third_position(2,1,3,lines),"between_first_second")
        self.assertEqual(third_position(2,1,4,lines),"immediately_behind_first")
        self.assertEqual(third_position(3,1,4,lines),"further_behind_first")
        self.assertEqual(third_position(5,1,4,lines),"immediately_behind_second")
        self.assertEqual(third_position(7,1,4,lines),"neither_line")
        self.assertEqual(third_role(3,lines),"role_third_or_later")
        self.assertEqual(third_role(2,lines),"role_second")
        self.assertEqual(third_role(6,lines),"role_second" if False else "role_leader")

    def test_scope_and_outcome_leakage_blocked(self):
        card=fixture()
        card["meeting_grade"]="G3"
        with self.assertRaises(ValueError):
            infer_triples(card,FIRST,SECOND,THIRD)
        card=fixture()
        card["riders"][0]["finish_position"]=1
        with self.assertRaises(ValueError):
            infer_triples(card,FIRST,SECOND,THIRD)

    def test_model_coefficients_and_research_cohort_frozen(self):
        self.assertEqual(THIRD["fit_races"],14081)
        self.assertEqual(THIRD["fit_candidate_rows"],69300)
        self.assertEqual(len(THIRD["coefficient_names"]),13)
        self.assertAlmostEqual(THIRD["coefficients"]["delta"],0.4891292239397931,places=10)
        self.assertEqual(THIRD["old_v7_third_coefficients_reused"],False)

if __name__=="__main__":
    unittest.main()
