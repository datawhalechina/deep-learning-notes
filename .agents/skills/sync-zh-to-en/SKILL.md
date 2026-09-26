---
name: sync-zh-to-en
description: Synchronize Chinese Quarto documentation into its English counterpart. Use when the user says "sync zh to en" or asks to update matching files under zh/ and en/ from the Chinese source.
---

# Synchronize Chinese Quarto Content to English

Treat the requested `zh/` content as authoritative and produce an accurate, complete English counterpart without changing unrelated work.

## Establish the Scope

Start with `git status --short`. Resolve each source path from `zh/<relative-path>` to `en/<relative-path>`. If the user names a file, directory, commit, or diff, sync only that scope. Do not interpret a focused request as permission to translate the entire repository.

Read both versions before editing. Inventory referenced figures, Mermaid sources, includes, citations, and imported repository helpers. Preserve unrelated target-only corrections and configuration.

## Synchronize the Content

- Match the Chinese document's front matter, heading hierarchy, paragraphs, lists, callouts, tables, footnotes, citations, code fences, equations, and figure order.
- Translate prose, headings, captions, alt text, and explanatory labels into natural technical English. Do not summarize or omit material.
- Preserve formulas, citation keys, URLs, anchors, identifiers, API names, code behavior, and expected outputs. Keep code structurally identical unless an English-specific path or an accompanying source fix requires a change.
- Preserve the English page's language-specific title and description while updating them to reflect the Chinese source. Retain legitimate target-only metadata such as authorship unless the user requests otherwise.
- Mirror new or renamed local assets into the corresponding `en/` figure directory. When editable diagram sources such as `.mmd` files exist, keep them paired with generated assets and translate visible labels while preserving layout and styling.
- If the source now relies on changed shared Python helpers, synchronize the required helper and call sites only when they are part of the requested change; otherwise report the dependency clearly.

## Validate the Result

Compare source and target structure: headings, fenced blocks, display math, citations, figure references, and footnotes should have matching coverage. Confirm every relative asset exists and inspect the English file for accidental Chinese prose. Review the final diff for unintended rewrites.

Render the affected English page or the HTML profile when practical. Run any required Python checks through the repository's `uv` environment. If Python files changed, run Ruff formatting and linting on those files plus focused pytest coverage. Always finish with `git diff --check` and state exactly which rendering and tests were completed.
