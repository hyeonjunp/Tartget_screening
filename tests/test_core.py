import unittest
from unittest.mock import patch

from app.core import apply_mutations, fold_decision, search_pdb, sequence_identity


class MutationTests(unittest.TestCase):
    def test_valid_combined_mutations(self):
        self.assertEqual(apply_mutations("ACDEFG", "C2A/F5W"), "AADEWG")

    def test_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "outside FASTA"):
            apply_mutations("ACD", "A4G")

    def test_reference_residue_mismatch(self):
        with self.assertRaisesRegex(ValueError, "not A"):
            apply_mutations("ACD", "A2G")

    def test_duplicate_position(self):
        with self.assertRaisesRegex(ValueError, "Multiple mutations"):
            apply_mutations("ACD", "A1G/A1V")


class FoldDecisionTests(unittest.TestCase):
    def test_strictly_separated_folds_pass(self):
        result = fold_decision([8, 9, 8.5, 10, 8.1], [7, 7.5, 6, 7, 7.2])
        self.assertEqual(result["status"], "passed")
        self.assertAlmostEqual(result["delta_robust"], 0.5)

    def test_equal_boundary_rejected(self):
        result = fold_decision([8, 9, 10, 11, 12], [7, 8, 6, 6, 6])
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["delta_robust"], 0)

    def test_missing_fold_fails_closed(self):
        self.assertEqual(fold_decision([8, 9, None, 10, 11], [1, 2, 3, 4, 5])["status"], "prediction_failed")

    def test_non_finite_fails_closed(self):
        self.assertEqual(fold_decision([8, 9, float("nan"), 10, 11], [1, 2, 3, 4, 5])["status"], "prediction_failed")


class SequenceIdentityTests(unittest.TestCase):
    def test_identity_and_difference(self):
        result = sequence_identity("ACDE", "ACNE")
        self.assertEqual(result["identity"], 0.75)
        self.assertEqual(result["differences"][0]["left_position"], 3)


class PdbApiMockTests(unittest.TestCase):
    @patch("app.core._request_json")
    def test_query_mutation_does_not_become_structure_annotation(self, request):
        request.side_effect = [
            {"result_set": [{"identifier": "1ABC"}]},
            {"data": {"entries": [{"rcsb_id": "1ABC", "struct": {"title": "EGFR"}, "exptl": [],
                "polymer_entities": [{"rcsb_entity_source_organism": [{"ncbi_scientific_name": "Homo sapiens", "rcsb_gene_name": [{"value": "EGFR"}]}],
                    "entity_poly": {"pdbx_seq_one_letter_code_can": "ACD"}, "rcsb_polymer_entity": {"pdbx_description": "EGFR"},
                    "rcsb_polymer_entity_container_identifiers": {"asym_ids": ["A"]}}]}]}}
        ]
        result = search_pdb("EGFR(L858R)")
        self.assertEqual(result[0]["chains"][0]["mutation_count"], None)
        self.assertIn("not inferred from the search term", result[0]["mutation_annotation"])
        search_payload = request.call_args_list[0].args[1]
        self.assertEqual(search_payload["query"]["parameters"]["value"], "EGFR")


if __name__ == "__main__":
    unittest.main()
