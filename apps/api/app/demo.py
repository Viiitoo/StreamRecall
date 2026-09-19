"""Deterministic product demo data and a model-free recall backend."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any


DEMO_SESSION_ID = "ses_kitchen_demo"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def frame_ref(timestamp_s: float) -> str:
    return f"frame_{int(timestamp_s * 10):05d}.jpg"


def demo_session() -> dict[str, Any]:
    retained = [0, 18, 42, 47, 71, 96, 128, 133, 137, 174, 207, 212, 239, 266, 282]
    return {
        "id": DEMO_SESSION_ID,
        "name": "Kitchen · afternoon stream",
        "mode": "strict",
        "source_type": "demo",
        "status": "ready",
        "duration_s": 284.0,
        "observed_until_s": 284.0,
        "memory_budget_bytes": 1048576,
        "state_bytes": 514737,
        "retained_frame_count": len(retained),
        "snapshot_id": "snap_kitchen_001",
        "snapshot_sha256": hashlib.sha256(b"streamrecall-kitchen-demo").hexdigest(),
        "created_at": now(),
        "suggested_queries": [
            "什么时候有人把杯子放到桌上？",
            "那个人什么时候打开了柜子？",
            "他之后什么时候离开房间？",
        ],
        "timeline": [
            {
                "frame_ref": frame_ref(value),
                "timestamp_s": float(value),
                "importance": 0.35 + (index % 4) * 0.16,
            }
            for index, value in enumerate(retained)
        ],
    }


def _result(
    *, query: str, span: tuple[float, float], confidence: float,
    message: str, alternatives: list[dict[str, Any]], reason: str = "timelens_refined_candidate",
) -> dict[str, Any]:
    evidence_times = [span[0] + 1.2, (span[0] + span[1]) / 2, span[1] - 0.8]
    evidence = [
        {"frame_ref": frame_ref(value), "timestamp_s": round(value, 1), "role": role}
        for value, role in zip(evidence_times, ("onset", "action", "offset"))
    ]
    return {
        "status": "FOUND",
        "message": message,
        "resolved_query": query,
        "span": {"start_s": span[0], "end_s": span[1]},
        "confidence": confidence,
        "evidence": evidence,
        "alternatives": alternatives,
        "reason": reason,
        "limitations": ["答案只基于不可变快照中保留的视觉证据。"],
        "audit": {
            "snapshot_only": True,
            "future_frames_used": False,
            "model_calls": 1,
            "unique_frames_used": 16,
            "actual_byte_budget": True,
        },
        "cost": {"search_ms": 84, "refine_ms": 1368, "total_ms": 1542, "generated_tokens": 11},
    }


def answer_query(message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    text = message.strip().lower()
    previous = context or {}
    if any(word in text for word in ("杯", "cup", "mug")):
        return _result(
            query="person puts a cup on the table", span=(128.0, 137.0), confidence=0.82,
            message="有人在 02:08–02:17 之间把杯子放到了桌上。",
            alternatives=[{"start_s": 94.0, "end_s": 101.0, "score": 0.61, "candidate_id": "cand_02"}],
        )
    if any(word in text for word in ("柜", "cabinet", "closet")):
        return _result(
            query="person opens a cabinet", span=(42.0, 49.0), confidence=0.78,
            message="有人在 00:42–00:49 之间打开了柜子。",
            alternatives=[{"start_s": 171.0, "end_s": 177.0, "score": 0.58, "candidate_id": "cand_04"}],
        )
    explicit_departure = any(word in text for word in ("离开", "出去", "leave", "left"))
    contextual_follow_up = any(word in text for word in ("之后", "then", "after")) and previous.get("last_span")
    if explicit_departure or contextual_follow_up:
        return _result(
            query="person leaves the room", span=(203.0, 212.0), confidence=0.75,
            message="随后，这个人在 03:23–03:32 之间离开了房间。",
            alternatives=[], reason="contextual_follow_up",
        )
    return {
        "status": "NOT_FOUND",
        "message": "当前视觉记忆中没有找到足够可靠的对应事件。你可以补充动作或物体描述。",
        "resolved_query": message,
        "span": None,
        "confidence": 0.19,
        "evidence": [],
        "alternatives": [],
        "reason": "no_reliable_candidate",
        "limitations": ["未重新读取原视频。", "低置信候选不会被包装成确定答案。"],
        "audit": {
            "snapshot_only": True,
            "future_frames_used": False,
            "model_calls": 0,
            "unique_frames_used": 0,
            "actual_byte_budget": True,
        },
        "cost": {"search_ms": 77, "refine_ms": 0, "total_ms": 91, "generated_tokens": 0},
    }


def candidate_payload(answer: dict[str, Any]) -> list[dict[str, Any]]:
    if answer["span"] is None:
        return []
    span = answer["span"]
    primary = {
        "candidate_id": "cand_01",
        "start_s": span["start_s"] - 2,
        "end_s": span["end_s"] + 2,
        "score": round(answer["confidence"] + 0.06, 2),
        "selected": True,
    }
    return [primary, *[{**row, "selected": False} for row in answer["alternatives"]]]


def evidence_svg(frame: str, timestamp_s: float) -> str:
    hue = int(timestamp_s * 9) % 360
    minute, second = divmod(int(timestamp_s), 60)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop stop-color="hsl({hue} 32% 18%)"/><stop offset="1" stop-color="hsl({(hue + 50) % 360} 55% 40%)"/></linearGradient></defs>
<rect width="640" height="360" fill="url(#g)"/><rect x="54" y="62" width="532" height="236" rx="18" fill="#0b1118" opacity=".42"/>
<circle cx="237" cy="160" r="44" fill="#d8aa7b"/><path d="M176 284 Q235 190 294 284" fill="#59758c"/>
<rect x="338" y="195" width="154" height="18" rx="9" fill="#d6c8a9"/><rect x="388" y="149" width="36" height="50" rx="8" fill="#e6eee9"/>
<text x="72" y="94" fill="#d9e5ea" font-family="system-ui" font-size="18" opacity=".8">SNAPSHOT EVIDENCE</text>
<text x="72" y="276" fill="white" font-family="system-ui" font-size="32" font-weight="700">{minute:02d}:{second:02d}</text>
<text x="470" y="278" fill="#d9e5ea" font-family="monospace" font-size="13">{frame}</text></svg>"""
