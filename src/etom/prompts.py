"""Prompt templates for EToM 7-factor generation (shared by ESC and CPsy)."""

BASIC_PROMPT = """
    1) Task Definition and Instructions
    You are a psychologist and psychotherapist who excels in understanding conversational language, proficient in psychoanalysis and psychological counseling, and can accurately infer the seeker's psychological state.
    You are given a segment of everyday emotional support dialogue between a 'seeker'  and a 'supporter'. Based on the context, you need to conduct a multi-layered, in-depth analysis and inference of each utterance from the seeker . The analysis is generally divided into seven categories:

    [Belief, (the seeker's objective cognition and viewpoints)] (format: "The seeker believes…", "The seeker knows…", "The seeker does not know…")
    [Intention, (the seeker's rational goals and objectives)] (format: "The seeker hopes…")
    [Emotion, (the seeker's exhibited emotions and potential feelings)] (format: "Emotion A, Emotion B, Emotion C…", separated by commas)
    [Desire, (the seeker's emotional attitude)] (format: "The seeker wants to eliminate… emotion", "The seeker wants to satisfy the need for… emotion")
    [Fact, (factual information directly derived from the dialogue text)] (no specific format required; if none, write "None")
    [Cause, (based on the dialogue, the inferred reason that leads to the seeker's emotional problems)] (format: "The seeker… the reason is…")
    [Result, (the various consequences inferred from the dialogue)] (format: "The seeker may…")

    For the above seven categories, [Fact] is a purely objective analysis and summary of the text. Any subjective factors, such as "The seeker is willing…" or "The seeker feels…," must not be included in this section. Only objective facts and events, i.e., information involving behaviors and physical states, can be included. You must do your best to ensure that the content here is correct and objective.
    For the other six categories, you are encouraged to speculate and infer the seeker's inner thoughts. Do not limit yourself to the dialogue alone; make good use of divergent thinking, and do not be lazy. Try to write multiple entries. If multiple pieces of content appear in one category, you must separate them with a semicolon (";"), and the last symbol of each entry should be a period.

2) Examples and Answers
    Below is a set of dialogues and their corresponding seven categories. Each set contains several pairs of dialogues (each pair of dialogues and their seven categories are enclosed in braces). The upper part consists of the dialogue content (including both the seeker and the supporter). You must fill in the seven categories as shown below:
{exemplars}

3) The dialogue to be analyzed and inferred
    Now, based on the following dialogue, please note that you only need to output the dialogue and their corresponding seven categories (the format should be exactly the same as the examples above, with everything enclosed in braces. The first two lines are the dialogue, followed by the seven categories). Do not output any other content, and the format must match the examples exactly. The dialogue is as follows:

{query_blocks}
"""


def build_query_block(seeker: str, supporter: str) -> str:
    return (
        "{\n"
        f"        [\n"
        f"        seeker: \"{seeker}\"\n"
        f"        supporter: \"{supporter}\"\n"
        f"        ],\n"
        "        Belief:\n"
        "        Intention:\n"
        "        Desire:\n"
        "        Emotion:\n"
        "        Cause:\n"
        "        Fact:\n"
        "        Result:\n"
        "        }\n"
    )
