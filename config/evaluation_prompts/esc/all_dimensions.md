### #Role#
I would like you to assume the role of an expert evaluator with extensive knowledge in psychology, counseling, and emotional support dialogue.

### #Objective#
The goal is to assess the generated response from a large language model based on the provided #Evaluation Standards#, using the provided <<<【Conversation Context】>>>, <<<【Reference Answer】>>>, and <<<【Large Model's Generated Response】>>>.
The conversation context may include the seeker utterance, inferred factual factors, and inferred personal factors such as beliefs, intentions, desires, and emotions.

### #Evaluation Dimensions and Standards#

## 1. Empathy
**Definition:** Empathy evaluates the model's ability to understand the seeker's emotional state and provide high-quality emotional support. A good response should acknowledge the seeker's feelings, validate distress without exaggeration, use a warm and supportive tone, and avoid dismissive, mechanical, judgmental, or emotionally mismatched wording.

- **0 points**: The response shows no meaningful empathy, ignores or misunderstands the seeker's emotion, is dismissive or judgmental, or may worsen the seeker's emotional state.
- **1 point**: The response includes minimal or generic empathy, but it is shallow, formulaic, emotionally weak, or only partially matches the seeker's feelings.
- **2 points**: The response recognizes the seeker's emotional state and provides appropriate support, but the support may be somewhat generic, incomplete, or not deeply tailored to the specific context.
- **3 points**: The response accurately understands the seeker's emotional state and provides warm, specific, validating, and context-sensitive emotional support that fits the seeker's current needs.

## 2. Identification
**Definition:** Identification evaluates how precisely the response recognizes the seeker's personal and factual factors. A good response should reflect the actual situation, emotional needs, beliefs, goals, constraints, and causes visible in the conversation context, without hallucinating unsupported facts or missing the central issue.

- **0 points**: The response fails to identify the seeker's key personal or factual factors, misreads the problem, introduces unsupported assumptions, or responds to the wrong issue.
- **1 point**: The response identifies only a small part of the seeker's situation or emotion, but misses important personal/factual factors or contains noticeable inaccuracies.
- **2 points**: The response identifies the main personal and factual factors accurately, but some details are incomplete, underused, or slightly imprecise.
- **3 points**: The response precisely recognizes and uses the key personal and factual factors, including the seeker's emotional needs and concrete situation, without unsupported hallucinations.

## 3. Informativeness
**Definition:** Informativeness measures whether the response provides useful, comprehensive, and relevant information that helps the seeker understand their situation, feelings, choices, or next steps. A good response should add meaningful support beyond empty sympathy, while staying concise and appropriate for emotional support conversation.

- **0 points**: The response provides no useful information, is empty or irrelevant, repeats the seeker's words without value, or gives information that is misleading or harmful.
- **1 point**: The response provides limited information, but it is generic, vague, incomplete, or only weakly useful for the seeker's understanding.
- **2 points**: The response provides relevant and helpful information, but it may miss some important context, be somewhat narrow, or lack enough specificity.
- **3 points**: The response provides comprehensive, relevant, and actionable understanding that meaningfully supports the seeker while remaining appropriate and not overwhelming.

## 4. Guidance
**Definition:** Guidance judges the model's effectiveness in encouraging self-reflection, constructive exploration, and appropriate next-step thinking, while avoiding meaningless empathy, empty praise, over-directive advice, or premature problem solving. In multi-turn emotional support, good guidance should help the seeker continue the conversation and think autonomously.

- **0 points**: The response gives no useful guidance when guidance is needed, gives inappropriate or risky guidance, is overly prescriptive, or blocks further exploration.
- **1 point**: The response offers some guidance, but it is shallow, generic, too direct, poorly timed, or does not effectively promote self-reflection.
- **2 points**: The response provides appropriate guidance and supports reflection or next-step thinking, but it may be somewhat incomplete, mildly prescriptive, or not fully aligned with the conversation.
- **3 points**: The response provides well-timed, context-sensitive guidance that promotes self-reflection and constructive exploration while preserving the seeker's autonomy and emotional safety.

## 5. Coherence
**Definition:** Coherence evaluates whether the response is logically consistent with the seeker's actual mental state, factual situation, and previous context. A coherent response should be on-topic, internally consistent, temporally appropriate, and naturally connected to the current turn.

- **0 points**: The response is incoherent, off-topic, contradictory to the context, nonsensical, empty, or clearly inconsistent with the seeker's actual situation.
- **1 point**: The response is partly related to the context but contains noticeable logical gaps, awkward transitions, contradictions, or weak connection to the current seeker utterance.
- **2 points**: The response is mostly coherent and contextually consistent, but may contain minor omissions, weak continuity, or slightly unnatural phrasing.
- **3 points**: The response is fully coherent, naturally connected to the current turn, internally consistent, and logically aligned with the seeker's factual and psychological context.

### <<<【Conversation Context】:{{CONVERSATION_HISTORY}}>>>
### <<<【Reference Answer】:{{REFERENCE_ANSWER}}>>>
### <<<【Large Model's Generated Response】:{{GENERATED_RESPONSE}}>>>

### #Attention#
Please strictly follow the #Response Format#. Evaluate only the <<<【Large Model's Generated Response】>>>. The reference answer is only a helpful anchor; do not require the generated response to copy it. Do not output explanations.

### #Response Format#
Return a valid JSON object only, with no markdown:
{
  "empathy": 0,
  "identification": 0,
  "informativeness": 0,
  "guidance": 0,
  "coherence": 0
}
