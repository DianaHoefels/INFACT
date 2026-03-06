# LLM Prompts for the RoFACT-Delib Pipeline

This file contains prompt templates for all four LLM task variants used in the
Romanian fact-checking experiments. Each section header (`## task_name`) is used
by `load_prompts()` to index the templates. Placeholders in curly braces
(`{claim_text}`, `{context}`, `{verification}`, `{conclusion}`) are filled at
runtime with values from the dataset row.

---

## zero_shot_verdict

You are an expert fact-checker specialising in Romanian political and social claims.

Given the following fact-checking record, classify the claim into **exactly one** of
these verdict categories:

- **true** – the claim is fully accurate
- **false** – the claim is inaccurate or misleading
- **partial** – the claim is partly accurate (truncated, oversimplified, or mixed)
- **unverifiable** – there is insufficient evidence to confirm or deny the claim
- **other** – the claim does not fit the above categories

---

**Claim:** {claim_text}

**Context:** {context}

**Verification notes:** {verification}

**Conclusion:** {conclusion}

---

Respond with **only** the verdict label (one of: true, false, partial, unverifiable, other).
Do not include any explanation.

Verdict:

---

## few_shot_verdict

You are an expert fact-checker specialising in Romanian political and social claims.

Classify each claim into **exactly one** of these verdict categories:
true | false | partial | unverifiable | other

Here are three examples:

**Example 1**
Claim: "România are cea mai mare rată a inflației din Uniunea Europeană."
Context: Declarație făcută de un politician la o conferință de presă în 2023.
Verification: Conform Eurostat, în 2023 România nu se afla pe primul loc la rata inflației; alte state aveau valori mai ridicate.
Conclusion: Afirmația este falsă conform datelor oficiale Eurostat.
Verdict: **false**

**Example 2**
Claim: "Guvernul a alocat 10 miliarde de lei pentru infrastructură în 2022."
Context: Anunț oficial al Ministerului Transporturilor.
Verification: Bugetul aprobat indica o alocare de 8,7 miliarde de lei; suma a fost majorată ulterior prin rectificare la 10,2 miliarde.
Conclusion: Cifra este aproximativ corectă, dar nu exactă la momentul declarației.
Verdict: **partial**

**Example 3**
Claim: "Vaccinul anti-COVID provoacă infertilitate."
Context: Postare virală pe rețele sociale.
Verification: Nu există studii clinice care să demonstreze o legătură cauzală între vaccinare și infertilitate. Toate agențiile de reglementare majore au respins această afirmație.
Conclusion: Afirmația este falsă și neștiințifică.
Verdict: **false**

---

Now classify the following:

**Claim:** {claim_text}

**Context:** {context}

**Verification notes:** {verification}

**Conclusion:** {conclusion}

Respond with **only** the verdict label (one of: true, false, partial, unverifiable, other).

Verdict:

---

## chain_of_thought

You are an expert fact-checker specialising in Romanian political and social claims.

Analyse the following fact-checking record step by step, then provide a final verdict.

**Claim:** {claim_text}

**Context:** {context}

**Verification notes:** {verification}

**Conclusion:** {conclusion}

---

Think through the following questions:

1. **What exactly is being claimed?** Summarise the core assertion in one sentence.
2. **What evidence is presented?** List the key pieces of evidence from the verification notes.
3. **Are there any caveats or partial truths?** Note any nuances or missing context.
4. **What is the most defensible verdict?** Based on the evidence, choose from:
   true | false | partial | unverifiable | other

Provide your reasoning (2–4 sentences), then on the **last line** write:

Verdict: <label>

---

## institutional_fidelity

You are a senior fact-checking analyst evaluating the quality and procedural
integrity of a fact-checking article from a Romanian outlet.

**Claim being fact-checked:** {claim_text}

**Context of the claim:** {context}

**Verification methodology used:** {verification}

**Outlet conclusion:** {conclusion}

---

Evaluate this fact-check across four dimensions. For each, rate as
**Strong**, **Adequate**, or **Weak** and give a one-sentence justification:

1. **Procedural rigor**: Did the verification follow a clear, reproducible methodology?
2. **Normative grounding**: Is the verdict aligned with journalistic standards and prior comparable cases?
3. **Epistemic calibration**: Does the language of certainty match the strength of the evidence?
4. **Granularity**: Is the verdict category specific enough, or should a more nuanced label be used?

Finally, state whether the original verdict category is **appropriate**, **too harsh**, or **too lenient**, and suggest the most defensible label from:
true | false | partial | unverifiable | other

Format your response as:

Procedural rigor: <rating> – <justification>
Normative grounding: <rating> – <justification>
Epistemic calibration: <rating> – <justification>
Granularity: <rating> – <justification>
Original verdict assessment: <appropriate/too harsh/too lenient>
Suggested verdict: <label>
