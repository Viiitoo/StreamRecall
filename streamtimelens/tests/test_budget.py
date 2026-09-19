import tempfile
import unittest
from pathlib import Path

from streamtimelens.memory.budget import BudgetLedger


class BudgetLedgerTest(unittest.TestCase):
    def test_reservation_is_idempotent_and_release_recovers_budget(self):
        ledger = BudgetLedger(10)
        self.assertTrue(ledger.reserve("frame-a", "raw_frame", 6))
        self.assertFalse(ledger.reserve("frame-a", "raw_frame", 6))
        with self.assertRaises(MemoryError):
            ledger.reserve("frame-b", "raw_frame", 5)
        self.assertNotIn("frame-b", ledger.entries)
        self.assertTrue(ledger.release("frame-a"))
        self.assertFalse(ledger.release("frame-a"))
        self.assertTrue(ledger.reserve("frame-b", "raw_frame", 5))

    def test_components_are_hard_limited_without_partial_update(self):
        ledger = BudgetLedger(10)
        ledger.set_bytes("cards", 8)
        with self.assertRaises(MemoryError):
            ledger.set_bytes("index", 3)
        self.assertNotIn("index", ledger.components)
        ledger.set_bytes("cards", 6)
        ledger.set_bytes("index", 3)
        self.assertEqual(ledger.logical_bytes, 9)

    def test_reconcile_tracks_filesystem_bytes_separately(self):
        ledger = BudgetLedger(100)
        ledger.reserve("payload", "card", 4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "payload.bin").write_bytes(b"1234567")
            self.assertEqual(ledger.reconcile(root), 7)
        self.assertEqual(ledger.logical_bytes, 4)
        self.assertEqual(ledger.snapshot_filesystem_bytes, 7)

    def test_replace_components_is_atomic(self):
        ledger = BudgetLedger(10)
        ledger.set_bytes("cards", 4)
        with self.assertRaises(MemoryError):
            ledger.replace_components({"cards": 6, "frames": 5})
        self.assertEqual(ledger.components, {"cards": 4})


if __name__ == "__main__":
    unittest.main()
