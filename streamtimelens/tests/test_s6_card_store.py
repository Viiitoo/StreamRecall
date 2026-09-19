import unittest

from streamtimelens.memory.card_store import CardStore
from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.memory.raw_cache import RawFrameCache
from streamtimelens.protocol.types import FramePacket


def card(card_id, start, end, **kwargs):
    return EvidenceCard(card_id, 0, start, end, card_id, **kwargs)


class CardStoreTest(unittest.TestCase):
    def test_crud_neighbors_and_round_trip_have_stable_temporal_order(self):
        store = CardStore()
        for item in (card("c", 2, 3), card("b", 1, 3), card("a", 1, 2)):
            store.insert(item)
        self.assertEqual(store.ids, ("a", "b", "c"))
        self.assertEqual(
            tuple(value.id if value else None for value in store.temporal_neighbors("b")),
            ("a", "c"),
        )
        restored = CardStore.from_rows(reversed(store.rows()))
        self.assertEqual(restored.ids, store.ids)
        self.assertEqual(restored.rows(), store.rows())
        self.assertEqual(store.delete("b").id, "b")
        self.assertNotIn("b", store)

    def test_accounting_separates_embedding_refs_and_releases_raw_owner(self):
        raw = RawFrameCache()
        packet = FramePacket(1.0, 1, b"jpeg", 1, 1, video_id="v")
        raw.add(packet, owner="card:a")
        item = card(
            "a", 1, 1, raw_ref_ids=["000000001.jpg"],
            raw_ref_status={"000000001.jpg": "available"},
            text_embedding={
                "format_version": 1, "dtype": "fp16", "length": 1,
                "scale": None, "data_b64": "ADw=",
            },
        )
        store = CardStore(raw)
        store.insert(item)
        accounting = store.accounting()
        self.assertGreater(accounting.card_payload_bytes, 0)
        self.assertGreater(accounting.embedding_bytes, 0)
        self.assertGreater(accounting.raw_ref_index_bytes, 0)
        self.assertEqual(accounting.total_bytes, len(
            __import__("json").dumps(
                item.serializable(), sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ).encode("utf-8")
        ) + 1 + accounting.temporal_index_bytes)
        store.delete("a")
        self.assertNotIn("000000001.jpg", raw)


if __name__ == "__main__":
    unittest.main()
