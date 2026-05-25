"""LLM-based outdated-memory judgment (ORM helper).

Replaces IJCAI/Update/discriminate_outdated.py. Uses the unified LLMClient
(role="judge", which is the Qwen2.5-32B-Instruct default).
"""

from __future__ import annotations

from src.llm_client import default_client


_PROMPT = """
1) Task Definition and Instructions
You are a highly skilled emotional support specialist, well-versed in the nuances of conversational language and expert at psychoanalysis within the context of emotional support dialogues. Your task is to determine whether a given piece of information from the previous conversation has become outdated in light of the current seeker's utterance and mental state during an ongoing emotional support conversation.

In an emotional support conversation, the flow can be dynamic. Topics may shift, and the seeker's emotions can change rapidly. We define outdated information as that which no longer aligns with the current topic, emotional state, or underlying intentions of the seeker. Consider:
- Topic Relevance
- Emotional Consistency
- Intentional Alignment

Answer with either "Relevant, keep in memory" or "Outdated, remove from memory".

2) Examples and Answers
Example 1:
Previous memory: "The seeker intends to resent the car owner that killed his cat yesterday."
Previous conversation history: [
    {"seeker": "My cat was killed by a car on the road yesterday. I can't stop crying. I feel so alone.", "supporter": "I'm sorry about what happened to you. May you tell me what you're coping with?"}
]
Current seeker's utterance: "I just remembered all the happy times we had together. I'm starting to feel a bit better now."
Current mind state: {"Belief": ["..."], "Intention": ["The seeker intends to focus on the positive memories of their pet."], "Emotion": ["A glimmer of hope, nostalgia."]}
Answer: Outdated, remove from memory

Example 2:
Previous memory: "The seeker is worried about an upcoming exam and is stressed about not having enough time to study."
Previous conversation history: [
    {"seeker": "I have an exam next week and I'm so stressed.", "supporter": "Let's make a study plan."}
]
Current seeker's utterance: "I've managed to create a study plan, but I'm still not sure if it's enough."
Current mind state: {"Belief": ["The seeker believes that their study plan may not be sufficient."], "Emotion": ["Anxiety, Uncertainty."]}
Answer: Relevant, keep in memory

3) The situation to be analyzed
Output exactly one of "Relevant, keep in memory" or "Outdated, remove from memory" and nothing else.
"""


def discriminate_outdated(
    outdated_memory: str,
    prev_conversation_history: str,
    current_seeker_utterance: str,
    current_mind_state: str,
) -> str:
    user_msg = (
        f"Previous memory: {outdated_memory}\n"
        f"Previous conversation history: {prev_conversation_history}\n"
        f"Current seeker's utterance: {current_seeker_utterance}\n"
        f"Current mind state: {current_mind_state}\n"
    )
    return default_client().chat(
        system=_PROMPT, user=user_msg, role="judge",
        temperature=0.0, max_tokens=64,
    )
