"""
PATTERN: PromptTemplate + LLM + StrOutputParser
DAY: 1
═══════════════════════════════════════════════

WHAT THIS PATTERN IS:
...

WHEN YOU'D USE THIS:
...

WHAT PROBLEM IT SOLVES:
...

THE FLOW:
input → PromptTemplate → LLM → StrOutputParser → plain string output

CONSTRUCTS INTRODUCED TODAY:
- PromptTemplate
- ChatOpenAI
- StrOutputParser
- pipe operator |
"""

# ── code below ──────────────────────────────────────────


# POST-ANALYSIS
# ─────────────────────────────────────────────────────────
# PromptTemplate.from_template():  ...
# .invoke():                       ...
# Why StrOutputParser exists:      ...