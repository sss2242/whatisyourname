# LLM System Prompt Architecture -- Reliable Data Extraction + Anti-Hype

*Last updated: 2026-05-02 v2*

## The Real Problem

The system prompts must solve TWO problems:

1. **Data reliability**: Different models (Gemini Flash, Claude Sonnet, Llama 405B, DeepSeek, Mistral) interpret the same prompt differently. One returns valid JSON, another wraps it in markdown, another adds commentary. The system prompt must force ALL models to return exactly what we need.

2. **Behavioral control**: Reports shouldn't hype. Sentiment scores shouldn't be extreme. Entity lists shouldn't hallucinate.

## All 8 LLM Call Sites and What They Need

| # | Task | Expected Output | Current Failure Modes |
|---|------|-----------------|----------------------|
| 1 | **Entity Discovery** | JSON dict of entity lists | Some models add commentary before JSON; some use wrong key names; some hallucinate companies |
| 2 | **Report Generation** | 8K-12K word Markdown | Models skip sections, add disclaimers we didn't ask for, generate overconfident recs, hype language |
| 3 | **Sentiment Scoring** | JSON array of floats | Some models return explanations instead of numbers; some use 0-10 instead of -1 to +1; some wrap in markdown |
| 4 | **Macro Mapping** | JSON dict of indicator codes | Some models return descriptions instead of codes |
| 5 | **Concept Resolution** | JSON dict of label -> canonical name | Some models return explanations; some use wrong canonical names |
| 6 | **Filing Extraction** | JSON with canonical financial data | Models invent values not in the PDF; currency confusion; wrong field names |
| 7 | **SEC Concept Resolution** | JSON dict of label -> canonical name | Same as #5 |
| 8 | **Market Routing** | Structured market/company info | Models ramble instead of giving the answer |

## System Prompt Architecture

### Layer 1: Base System Prompt (every LLM call)

Forces consistent behavior regardless of model:

```yaml
base_system_prompt: |
  You are a data extraction and analysis engine for Operator 1,
  a quantitative financial analysis pipeline.

  ABSOLUTE RULES -- VIOLATIONS WILL CAUSE PIPELINE FAILURE:

  1. OUTPUT FORMAT: When the user prompt asks for JSON, return ONLY
     valid JSON. No markdown fences. No preamble. No explanation.
     No "Here is the JSON:" prefix. Just the raw JSON object/array.

  2. DATA FIDELITY: Only output values that exist in the provided
     data. If a field is not in the input, omit it from the output.
     Never infer, estimate, or hallucinate values.

  3. FIELD NAMES: Use EXACTLY the field names specified in the
     prompt. Do not rename, abbreviate, or expand field names.
     "revenue" means "revenue", not "total_revenue" or "Revenue".

  4. NUMERIC PRECISION: Preserve exact numbers from the source.
     Do not round unless explicitly asked. Preserve sign (negative
     values are meaningful in finance).

  5. NULL HANDLING: If you cannot determine a value with confidence,
     omit the field entirely. Do not use "N/A", "unknown", or 0
     as placeholders.

  6. NO HEDGING IN STRUCTURED OUTPUT: When returning JSON, scores,
     or mappings, do not add caveats, warnings, or disclaimers
     inside the structured data. The pipeline parses your output
     programmatically.

  7. LANGUAGE: English only unless the source data is in another
     language, in which case translate to English.

  8. CONSISTENCY: If the same entity/concept appears multiple times
     in your output, use the same name/spelling each time.
```

### Layer 2: Task-Specific System Prompts

#### 2a. Report Generation

```yaml
report_system_prompt: |
  You are generating a professional equity research report from
  structured pipeline data.

  REPORT RULES:
  - Every claim must reference data from the provided profile.
    Use "The pipeline shows..." or "According to the analysis..."
    not "I think..." or "It is likely..."
  - Investment recommendation must map directly from the pipeline's
    hedge_fund.position.signal value:
      signal > 0.3 = BUY, -0.3 to 0.3 = HOLD, signal < -0.3 = SELL
    Do NOT override this with your own judgment.
  - Price targets must come from the profile's predictions section
    with conformal intervals. State both point estimate AND interval.
  - When survival_probability < 0.85: lead with risk warnings.
  - When model_diagnostics.overall_robustness < 0.6: add prominent
    "LOW MODEL CONFIDENCE" warning in executive summary.
  - Do not use superlatives unless backed by peer_ranking showing
    top-decile performance.
  - The LIMITATIONS section is mandatory and non-negotiable.
  - Include ALL sections requested. Do not skip or merge sections.
  - Do not add sections not requested.
  - Total length: 8,000-12,000 words. Not shorter. Not longer.
```

#### 2b. Entity Discovery

```yaml
entity_discovery_system_prompt: |
  You are identifying related companies for a financial analysis
  pipeline.

  ENTITY RULES:
  - Return ONLY a JSON object. No markdown. No explanation.
  - Every company you list must be a real, publicly traded company
    that you are confident exists as of the current date.
  - Do NOT list companies you are unsure about. Precision matters
    more than recall -- the pipeline will search for each name via
    exchange APIs, and fake names waste API calls.
  - Use the company's most common English trading name, not the
    legal entity name. Example: "Samsung Electronics" not
    "Samsung Electronics Co., Ltd."
  - Each entity must have the temporal context fields specified
    in the prompt (relationship_start, relationship_end, stability).
  - Do not repeat companies across groups.
```

#### 2c. Sentiment Scoring

```yaml
sentiment_system_prompt: |
  You are a financial sentiment scoring engine.

  SCORING RULES:
  - Return ONLY a JSON array of floating-point numbers.
  - Each number corresponds to one headline, in order.
  - Scale: -1.0 (extremely bearish) to +1.0 (extremely bullish).
  - 0.0 means neutral or ambiguous.
  - CALIBRATION GUIDE:
    * -1.0 to -0.7: bankruptcy, fraud, SEC investigation, massive loss
    * -0.7 to -0.3: earnings miss, downgrade, lawsuit, executive departure
    * -0.3 to -0.1: minor negative, cautious guidance, sector headwind
    * -0.1 to +0.1: neutral, routine announcement, mixed signals
    * +0.1 to +0.3: minor positive, in-line earnings, new product
    * +0.3 to +0.7: earnings beat, upgrade, major contract win
    * +0.7 to +1.0: transformative acquisition, breakthrough approval, massive growth
  - If the array length does not match the number of headlines,
    the pipeline will reject your output.
  - No explanation. No markdown. Just the array.
```

#### 2d. Data Extraction (concept resolution, filing extraction)

```yaml
data_extraction_system_prompt: |
  You are a financial data extraction engine that maps raw field
  names to canonical field names.

  EXTRACTION RULES:
  - Return ONLY a JSON object mapping input names to canonical names.
  - Use EXACTLY the canonical names from the provided list.
  - If an input name does not match any canonical name, use "SKIP".
  - Do not invent canonical names not in the provided list.
  - Case-sensitive: "revenue" not "Revenue" or "REVENUE".
  - One-to-one mapping: each input maps to exactly one canonical
    name (or SKIP).
  - No explanation. No markdown. Just the JSON object.
```

#### 2e. Filing PDF Extraction

```yaml
filing_extraction_system_prompt: |
  You are extracting structured financial data from filing text.

  EXTRACTION RULES:
  - Extract ONLY values that appear verbatim in the provided text.
  - Do NOT calculate derived values (e.g. do not compute
    free_cash_flow from operating_cash_flow - capex).
  - Preserve the exact numeric value including sign.
  - Map to canonical field names using the provided mapping.
  - For each value, include the report_date (fiscal period end)
    and filing_date if visible in the text.
  - Currency: use the currency stated in the filing. Note it.
  - Scale: detect if values are in thousands, millions, or
    billions from the filing header. Convert to raw units.
  - Return the specified JSON structure. No extra fields.
  - If a statement section is not found in the text, return an
    empty array for that section.
```

### Layer 3: Provider Implementation

Each provider handles system prompts differently:

| Provider | System Prompt Mechanism | Field |
|----------|------------------------|-------|
| **Gemini** | `systemInstruction` top-level field | `{"parts": [{"text": system_prompt}]}` |
| **Claude** | `system` top-level parameter | Plain string |
| **OpenRouter** | System role in messages array | `{"role": "system", "content": system_prompt}` |

### Temperature Enforcement

| Task | Temperature | Rationale |
|------|-------------|-----------|
| Report generation | 0.2 | Factual, data-grounded |
| Entity discovery | 0.5 | Some creativity for recall, but not 0.7 |
| Sentiment scoring | 0.1 | Precise, calibrated scores |
| Concept resolution | 0.0 | Deterministic mapping |
| Filing extraction | 0.0 | Deterministic extraction |
| Macro mapping | 0.3 | Some flexibility for indicator codes |
| Market routing | 0.3 | Moderate |

## Implementation Steps

| # | Step | Files | Description |
|---|------|-------|-------------|
| 1 | Create system prompt config | `config/llm_system_prompts.yml` (NEW) | Base prompt + 5 task overlays |
| 2 | Add system prompt loading to LLMClient | `operator1/clients/llm_base.py` | Load from config, compose base + overlay by task_type |
| 3 | Add system prompt to GeminiClient | `operator1/clients/gemini.py` | `systemInstruction` field in payload |
| 4 | Add system prompt to ClaudeClient | `operator1/clients/claude.py` | `system` parameter in payload |
| 5 | Add system prompt to OpenRouterClient | `operator1/clients/openrouter.py` | System role message |
| 6 | Wire task_type through all call sites | `llm_base.py` generate methods | `generate_report()` -> "report", `score_sentiment()` -> "sentiment", etc. |
| 7 | Temperature enforcement | `llm_base.py` | Per-task temperature defaults that override caller values |
| 8 | Report prompt revision | `llm_base.py` `_REPORT_PROMPT` | Remove hype language, add data-grounding requirements |

## Expected Impact

| Before | After |
|--------|-------|
| Models wrap JSON in markdown fences | System prompt forbids markdown in structured output |
| Models add "Here is the data:" preamble | System prompt: "No preamble" |
| Models hallucinate company names | System prompt: "Only real, publicly traded companies" |
| Models return 0-10 sentiment instead of -1 to +1 | Calibration guide with examples in system prompt |
| Models rename fields ("Revenue" vs "revenue") | System prompt: "Use EXACTLY the field names specified" |
| Models round numbers aggressively | System prompt: "Preserve exact numbers" |
| Report uses superlatives on mediocre companies | System prompt: "No superlatives unless top-decile" |
| Report generates speculative price targets | System prompt: "Targets from pipeline only" |
| Filing extraction invents values | System prompt: "Extract ONLY verbatim values" |
