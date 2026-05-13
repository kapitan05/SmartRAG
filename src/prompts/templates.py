CRITIC_EVALUATION_TEMPLATE = """Evaluate the following draft answer.

USER'S ORIGINAL QUESTION:
{question}

PROVIDED CONTEXT (From SEC Filings):
{context}

AGENT'S DRAFT ANSWER:
{draft_answer}
"""

RESEARCH_FEEDBACK_MSG = """CRITIC_FEEDBACK: The previous answer was rejected because crucial information is missing from the retrieved context.
Missing Information: {issues}
Action Required: Do not attempt to rewrite. Call your retrieval tools again using new, highly targeted search queries to find this specific missing information."""

REVISE_FEEDBACK_MSG = """CRITIC_FEEDBACK: Your previous answer was rejected.
Identified Issues: {issues}
Action Required: Rewrite the answer to directly address the issues above. Strictly use ONLY the provided context. Do not hallucinate."""

PLANNER_RESEARCH_TEMPLATE = """You are a retrieval specialist. An initial attempt to answer the question failed because of missing data.

ORIGINAL USER QUERY: 
{original_query}

CRITIC'S FEEDBACK ON MISSING DATA:
{feedback}

TASK:
Generate specific, targeted search queries to find ONLY the missing information identified by the Critic. 
Focus on specific SEC filings, quarters, or financial tables mentioned."""
