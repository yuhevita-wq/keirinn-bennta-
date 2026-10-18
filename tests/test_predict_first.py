"""Offline checks of adopted independent first-place inference."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predict_first import (ALL_FEATURES, BASE_FEATURES, STRUCT_FEATURES,
                           build_features, parse_lines, predict, require_scope)

MANIFEST = json.loads(
    (ROOT / "models/first_win_2026sep_manifest.json").read_text(encoding="utf-8")
)


def fixture():
    riders = []
    for i in range(1, 8):
        riders.append({
            "car_no": i,
            "score": 101.5 - 1.8 * i,
            "age": 27 + 2 * i,
            "term": 999 if i == 4 else 90 + 4 * i,
            "s_count": i % 3,
            "b_count": 8 if i in (1, 4, 6) else 1,
            "nige_count": 5 if i in (1, 4) else 0,
            "makuri_count": 2 if i in (1, 4, 6) else 0,
            "sashi_count": i % 4,
            "mark_count": i % 3,
            "first_count": 2 + i,
            "second_count": i % 4 + 1,
            "third_count": 1 + i % 3,
            "outside_count": 13,
            "win_rate": 5.0 + i * 3.1,
            "top2_rate": 22.0 + i * 2,
            "top3_rate": 31.0 + i * 2,
            "gear": 3.92,
        })
    return {
        "meeting_grade": "F1",
        "race_type": "Ｓ級予選",
        "predicted_line_formation": "1-2-3/4-5/6-7",
        "riders": riders,
    }


class FirstWinTests(unittest.TestCase):
    def test_dimensions_and_frozen_manifest(self):
        self.assertEqual(len(BASE_FEATURES), 17)
        self.assertEqual(len(STRUCT_FEATURES), 19)
        self.assertEqual(len(ALL_FEATURES), 37)
        self.assertEqual(MANIFEST["feature_names"], list(ALL_FEATURES))
        self.assertEqual(len(MANIFEST["model"]["coefficients"]), 74)
        self.assertEqual(MANIFEST["model"]["transform"]["std"][-1], 1.0)
        self.assertEqual(MANIFEST["train_max_date"], "2026-08-30")

    def test_probabilities_normalized(self):
        result = predict(fixture(), MANIFEST)
        forecasts = result["predictions"]
        self.assertEqual([r["car_no"] for r in forecasts], list(range(1, 8)))
        for key in ("raw_independent_win_probability", "calibrated_first_place_probability"):
            self.assertAlmostEqual(sum(r[key] for r in forecasts), 1.0, places=9)
            self.assertTrue(all(0 < r[key] < 1 for r in forecasts))

    def test_special_term_is_binary_category_not_ordinal_999(self):
        race = fixture()
        ordered = require_scope(race)
        lines = parse_lines(race["predicted_line_formation"], [c for c, _ in ordered])
        features = build_features(ordered, lines)
        self.assertIsNone(features[4][BASE_FEATURES.index("term")])
        self.assertEqual(features[4][-1], 1.0)
        self.assertEqual(features[1][-1], 0.0)

    def test_results_and_market_data_are_not_inputs(self):
        for field in ("result", "odds"):
            race = fixture()
            race[field] = {"example": 123}
            with self.assertRaises(ValueError):
                predict(race, MANIFEST)
        race = fixture()
        race["riders"][0]["finish_position"] = 1
        with self.assertRaises(ValueError):
            predict(race, MANIFEST)

    def test_other_grades_and_incomplete_lines_rejected(self):
        for race_type, grade in (("Ａ級予選", "F1"), ("Ｓ級準決勝", "G3")):
            race = fixture()
            race["race_type"] = race_type
            race["meeting_grade"] = grade
            with self.assertRaises(ValueError):
                predict(race, MANIFEST)
        race = fixture()
        race["predicted_line_formation"] = "1-2-3/4-5/6"
        with self.assertRaises(ValueError):
            predict(race, MANIFEST)

    def test_deterministic(self):
        race = fixture()
        self.assertEqual(predict(race, MANIFEST), predict(race, MANIFEST))


if __name__ == "__main__":
    unittest.main()
