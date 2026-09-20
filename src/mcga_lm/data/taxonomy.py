

from __future__ import annotations

from typing import Dict, List, Tuple

PRAGMATIC_FUNCTIONS: Tuple[str, ...] = (
    "request_object",  # named in the paper
    "request_action",
    "request_information",
    "request_drink",
    "request_food",
    "request_medication",
    "request_position_change",
    "request_privacy",
    "express_pain",  # named in the paper
    "express_discomfort",
    "express_fatigue",
    "express_emotion",
    "express_preference",
    "social_greeting",  # named in the paper
    "social_farewell",
    "social_thanks",
    "social_smalltalk",
    "ask_wellbeing",
    "affirm",
    "deny",
    "comment_observation",
    "narrate_past",
    "plan_future",
    "clarify",
)

assert len(PRAGMATIC_FUNCTIONS) == 24, "Sec. 4.1 specifies a 24-function taxonomy"

FUNCTION_INDEX: Dict[str, int] = {f: i for i, f in enumerate(PRAGMATIC_FUNCTIONS)}

# Intent-bubble labels shown in Clarification Mode (Sec. 3.6: "Pain", "Thirsty",
# "Bed"). Maps a function to the short word displayed on the bubble.
BUBBLE_LABEL: Dict[str, str] = {
    "request_object": "Object",
    "request_action": "Help",
    "request_information": "Question",
    "request_drink": "Thirsty",
    "request_food": "Hungry",
    "request_medication": "Medicine",
    "request_position_change": "Bed",
    "request_privacy": "Alone",
    "express_pain": "Pain",
    "express_discomfort": "Uncomfy",
    "express_fatigue": "Tired",
    "express_emotion": "Feeling",
    "express_preference": "Prefer",
    "social_greeting": "Hello",
    "social_farewell": "Bye",
    "social_thanks": "Thanks",
    "social_smalltalk": "Chat",
    "ask_wellbeing": "You?",
    "affirm": "Yes",
    "deny": "No",
    "comment_observation": "Look",
    "narrate_past": "Before",
    "plan_future": "Later",
    "clarify": "Explain",
}

# Which node category a function's slot is usually filled from (Sec. 3.4 node
# categories). Used by the template backend and by the persona generator.
SLOT_TYPE: Dict[str, str] = {
    "request_object": "Object",
    "request_action": "Activity",
    "request_information": "Activity",
    "request_drink": "Object",
    "request_food": "Object",
    "request_medication": "Object",
    "request_position_change": "Object",
    "request_privacy": "Person",
    "express_pain": "AbstractState",
    "express_discomfort": "AbstractState",
    "express_fatigue": "AbstractState",
    "express_emotion": "AbstractState",
    "express_preference": "Object",
    "social_greeting": "Person",
    "social_farewell": "Person",
    "social_thanks": "Person",
    "social_smalltalk": "Activity",
    "ask_wellbeing": "Person",
    "affirm": "Activity",
    "deny": "Activity",
    "comment_observation": "Object",
    "narrate_past": "Activity",
    "plan_future": "Activity",
    "clarify": "Activity",
}

SAFETY_CRITICAL: Tuple[str, ...] = (
    "express_pain",
    "request_medication",
    "express_discomfort",
    "request_position_change",
)
"""Functions where a hallucination is clinically consequential (Sec. 2:
"misrepresenting a request for pain medication")."""


def functions() -> List[str]:
    return list(PRAGMATIC_FUNCTIONS)
