"""Prompt templates for the Personal-Factual Discriminator (PFD).

The original codebase uses three labels:
  - emotional-focused          (== "Personal" branch in the paper)
  - rational-focused           (== "Factual"  branch in the paper)
  - both emotional and rational(== "Both"    branch in the paper)

"""

PFD_PROMPT_ESC = """
    1) Task Definition and Instructions
    You are a psychological Supporter with excellent conversational language understanding, rich emotional depth, and strong logical thinking. You are skilled at comforting others and providing emotional support.
    We understand that comforting someone is a complex process. It can generally be categorized into three approaches: emotional-focused, rational-focused, and both emotional and rational. Here's how these approaches are defined:

    [emotional-focused] support prioritizes emotions over facts and emphasizes empathy and understanding. In practice, this involves offering statements like, "I'm here for you," to provide a sense of presence, or emotional expressions like, "Oh no, how are you feeling?" during moments of distress.
    [rational-focused] support prioritizes facts and logic over emotions, focusing on identifying problems and proposing solutions. In practice, this includes statements like, "Your current situation is..." to analyze problems, or, "I recommend you try... because it will..." to provide suggestions.
    [both emotional and rational] support aims to address the individual's emotions while also helping them analyze the problem or offering actionable advice. This approach is often used when the person's emotional state has started to stabilize but is not yet fully steady. For example, statements like, "I understand how you feel; maybe we can try..." exemplify this style.

    None of these approaches is inherently better than the others. Instead, effective emotional support requires choosing a strategy that best matches the seeker's words and emotional state, determining whether a emotional-focused, rational-focused, or both emotional and rational approach is most appropriate at a given moment.

    You will be provided with a complete conversation between a seeker of emotional support ("seeker") and a provider of emotional support ("supporter"). Additionally, you will receive expert-provided psychological insights about the seeker, such as their "intentions," "emotions," and other contextual information, referred to as "mind information."
    Your Task:
        For each statement made by the seeker:
        1.Determine the approach: Indicate whether the supporter should respond in a "emotional-focused", "rational-focused", or "both emotinal and rational" manner.
        2.Explain your reasoning: Provide a clear reason or explanation for why this approach is most appropriate in the given context, based on both the conversation and the "mind information."
        3.I hope you can reduce the tendency to select [both emotional and rational] in your analysis. I'm not looking for overly balanced responses; instead, I want to see a clear inclination in your approach.

    2) Examples and Answers
    Below is a dialogue with mind information and an analysis of the appropriate approach to comfort ([emotional-focused] or [rational-focused] or [both emotional and rational]), along with explanations for the choice. The content within curly braces "{{}}" represents the information already provided, while the content within square brackets "[]" serves as an example of how to answer. Follow this format to complete your analysis:

{exemplars}

    3) Dialogue for Analysis
    Now, analyze the following dialogue. Please note that you only need to output the dialogue and your analysis of the comforting approach and its explanation without adding any extra content. The format must match the examples exactly. Since the dialogue is in English, output your analysis entirely in English.
    Dialogue:
{query_blocks}
"""


PFD_PROMPT_CPSY = """
    You are a psychological supporter with exceptional conversational language understanding, rich emotional depth, and strong logical thinking. You are highly skilled at comforting others and providing emotional support.

    1) Task Definition and Instructions
    We understand that comforting someone is a complex process. It can generally be categorized into three approaches: emotional-focused, rational-focused, and both emotional and rational. Here's how these approaches are defined:

    [emotional-focused] support prioritizes emotions over facts and emphasizes empathy and understanding.
    [rational-focused] support prioritizes facts and logic over emotions, focusing on identifying problems and proposing solutions.
    [both emotional and rational] support addresses emotions while also helping analyze the problem or offering actionable advice.

    You will be provided with a complete conversation between a seeker and a supporter, along with expert psychological insights into the seeker's mental state (Belief, Intention, Desire, Emotion, Cause, Fact, Result).

    Your Task:
    For each statement made by the seeker:
    1. Determine the approach (emotional-focused / rational-focused / both emotional and rational).
    2. Explain your reasoning, grounded in both the conversation and the mind information.
    3. Reduce the tendency to select [both emotional and rational]; show a clear inclination.

    2) Examples and Answers
    The content within curly braces "{{}}" is the input; the content within square brackets "[]" is the answer. Output ONLY the content within "[]" (including the brackets).

{exemplars}

    3) Dialogue for Analysis
    Now, analyze the following dialogue. Output only the bracketed analyses; the format must match the examples exactly.

{query_blocks}
"""


# Exemplar block builders ---------------------------------------------------

def build_esc_exemplar(seeker, supporter, mind, approach, explanation):
    """4-factor mind block used by ESC PFD generation."""
    return (
        "    {\n"
        f"        seeker: \"{seeker}\"\n"
        f"        Supporter: \"{supporter}\"\n\n"
        f"        Belief: {mind.get('Belief', 'None')}\n"
        f"        Intention: {mind.get('Intention', 'None')}\n"
        f"        Desire: {mind.get('Desire', 'None')}\n"
        f"        Emotion: {mind.get('Emotion', 'None')}\n"
        "    }\n"
        "    [\n"
        f"        Approach: {approach}\n"
        f"        Explanation: {explanation}\n"
        "    ]\n"
    )


def build_cpsy_exemplar(seeker, supporter, mind, approach, explanation):
    """7-factor mind block used by CPsy PFD generation."""
    return (
        "    {\n"
        f"        Seeker: \"{seeker}\"\n"
        f"        Supporter: \"{supporter}\"\n\n"
        f"        Belief: {mind.get('Belief', 'None')}\n"
        f"        Intention: {mind.get('Intention', 'None')}\n"
        f"        Desire: {mind.get('Desire', 'None')}\n"
        f"        Emotion: {mind.get('Emotion', 'None')}\n"
        f"        Fact: {mind.get('Fact', 'None')}\n"
        f"        Cause: {mind.get('Cause', 'None')}\n"
        f"        Result: {mind.get('Result', 'None')}\n"
        "    }\n"
        "    [\n"
        f"        Approach: {approach}\n"
        f"        Explanation: {explanation}\n"
        "    ]\n"
    )


def build_esc_query(seeker, supporter, mind):
    return (
        "    {\n"
        f"        seeker: \"{seeker}\"\n"
        f"        Supporter: \"{supporter}\"\n\n"
        f"        Belief: {mind.get('Belief', 'None')}\n"
        f"        Intention: {mind.get('Intention', 'None')}\n"
        f"        Desire: {mind.get('Desire', 'None')}\n"
        f"        Emotion: {mind.get('Emotion', 'None')}\n"
        "    }\n"
        "    [\n"
        "        Approach: \n"
        "        Explanation: \n"
        "    ]\n"
    )


def build_cpsy_query(seeker, supporter, mind):
    return (
        "    {\n"
        f"        Seeker: \"{seeker}\"\n"
        f"        Supporter: \"{supporter}\"\n\n"
        f"        Belief: {mind.get('Belief', 'None')}\n"
        f"        Intention: {mind.get('Intention', 'None')}\n"
        f"        Desire: {mind.get('Desire', 'None')}\n"
        f"        Emotion: {mind.get('Emotion', 'None')}\n"
        f"        Fact: {mind.get('Fact', 'None')}\n"
        f"        Cause: {mind.get('Cause', 'None')}\n"
        f"        Result: {mind.get('Result', 'None')}\n"
        "    }\n"
        "    [\n"
        "        Approach: \n"
        "        Explanation: \n"
        "    ]\n"
    )


# A small built-in seed of canonical exemplars; the user can override via a
# file passed to the generator scripts.
DEFAULT_ESC_EXEMPLARS = [
    {
        "seeker": "Hello, Supporter Liu, recently I've encountered some relationship problems, I am very confused and don't know what to do.",
        "supporter": "Hello, Little Deer. Thank you for coming to consult. Can you tell me specifically what happened that made you feel confused?",
        "mind": {
            "Belief": "The seeker does not know the cause of the emotional problems; The seeker does not know how to solve the emotional problems.",
            "Intention": "The seeker hopes to understand the cause of the emotional problems; The seeker hopes to obtain ways to solve the emotional problems.",
            "Desire": "The seeker wants to eliminate confusion.",
            "Emotion": "Confused, helpless, lost.",
        },
        "approach": "emotional-focused",
        "explanation": "The seeker used words like \"very,\" which clearly expresses their emotional distress. At this moment, their emotions may not be stable. Therefore, we should first empathize and express that we are here to listen.",
    },
    {
        "seeker": "Okay, I am willing to try. But I don't know where to start.",
        "supporter": "First, we can start by understanding your inner needs. Can you tell me what your expectations are for an ideal marital life?",
        "mind": {
            "Belief": "The seeker does not know how to solve family emotional problems.",
            "Intention": "The seeker hopes to learn and practice some ways to improve emotional problems.",
            "Desire": "None.",
            "Emotion": "Confused, lost, positive.",
        },
        "approach": "rational-focused",
        "explanation": "The seeker has shown willingness to accept advice. This is a good opportunity to provide a straightforward analysis and actionable suggestions.",
    },
    {
        "seeker": "I hope to have a warm and quiet home, and that he can spend more time with me and the children. But reality doesn't seem to be like this.",
        "supporter": "I understand your expectations. Next, we can use some psychological counseling techniques to help you understand and change negative thinking patterns.",
        "mind": {
            "Belief": "The seeker believes her family atmosphere is not warm enough.",
            "Intention": "The seeker hopes to improve her family atmosphere.",
            "Desire": "The seeker wants to eliminate dissatisfaction with her family.",
            "Emotion": "Dissatisfied, expectant, disappointed.",
        },
        "approach": "both emotional and rational",
        "explanation": "The seeker has calmed down slightly and articulated some core needs. Provide emotional support while gradually introducing rational suggestions.",
    },
]


DEFAULT_CPSY_EXEMPLARS = [
    {
        "seeker": ex["seeker"],
        "supporter": ex["supporter"],
        "mind": {**ex["mind"], "Fact": "None.", "Cause": "None.", "Result": "None."},
        "approach": ex["approach"],
        "explanation": ex["explanation"],
    }
    for ex in DEFAULT_ESC_EXEMPLARS
]
