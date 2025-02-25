# Fact Checking AI Agent using Tavily, Arxiv, Google Search

**Problem Statement**: Misinformation and Fake information are rampant online.

**Use Case**: Automated Fact-Checking News Articles

**Test Agent** [https://huggingface.co/spaces/rprav007/LangGraph_3_Tool_Research_Agent]

**Tools:**

1. Tavily - Get the claims and statements from the news article that needs to be verified
2. Arxiv - Search research papers and publications that support or refute the claims
3. Google Search - Match claims with reliable sources such as government, fact checking websites to connect the dots

**Workflow:**

**1. Claim Extraction:** The user provides the news article to the agent. Tavily analyzes the text and extracts specific claims that need to be fact-checked.  
**2. Evidence Search:** The agent uses Arxiv to find research papers related to the claims. It also uses Google Search to find reliable sources that can provide evidence.  
**3. Fact-Checking:** The agent compares the claims in the news article with the evidence found in research papers and reliable sources. It assesses the level of support or contradiction for each claim.  
**4. Report Generation:** The agent generates a report summarizing its findings, including:
  * The claims made in the news article.
  * The evidence found to support or refute each claim.
  * An assessment of the veracity of each claim (e.g., "True," "False," "Partially True," "Unverified").

**Test Cases:**

* Fact check - Drinking coffee can increase your risk of developing cancer
* Fact Check - Scientists have discovered a new planet that is capable of supporting life
* Fact Check - The government is secretly using 5G technology to control people's minds
=======
---
title: Obserch Midterm
emoji: 💻
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
license: mit
short_description: Obesity focussed AI agent
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
