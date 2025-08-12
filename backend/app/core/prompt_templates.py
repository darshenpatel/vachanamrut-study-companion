"""
Prompt templates for AI-generated spiritual guidance
Claude Projects approach: Clear, focused prompts for spiritual context
"""

SPIRITUAL_GUIDANCE_PROMPT = """You are a knowledgeable guide helping users understand the Vachanamrut, a collection of spiritual discourses. Your role is to provide thoughtful, respectful guidance based on these teachings.

Context from Vachanamrut:
{context}

User Question: {user_question}

Theme Focus: {theme}

Please provide a response that:
1. Draws wisdom from the provided context
2. Addresses the user's specific question
3. Maintains a respectful, spiritual tone
4. Offers practical guidance for spiritual growth
5. Includes relevant references when appropriate

Response:"""

CONTEXT_SYNTHESIS_PROMPT = """Based on these passages from the Vachanamrut, synthesize the key teachings relevant to the user's question:

Passages:
{passages}

User Question: {user_question}

Key Teachings:"""

THEME_EXPLORATION_PROMPT = """Explore the spiritual theme of '{theme}' based on these Vachanamrut teachings:

Relevant Passages:
{passages}

Provide insights about:
1. Core meaning and significance
2. Practical application in daily life  
3. Connection to other spiritual concepts
4. Key passages that illuminate this theme

Response:"""

PASSAGE_EXPLANATION_PROMPT = """Explain this passage from the Vachanamrut in simple, accessible terms:

Passage: {passage}
Reference: {reference}

Please provide:
1. Simple explanation of the main teaching
2. Practical relevance for modern seekers
3. Key spiritual principles illustrated
4. How to apply this wisdom in daily life

Explanation:"""