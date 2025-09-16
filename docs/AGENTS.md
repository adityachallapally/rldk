# Documentation Guidelines

These instructions apply to all files under `docs/` unless a deeper directory provides additional rules.

## Style & Formatting

- Prefer concise headings and keep front-matter (if present) at the top of the file.
- Wrap new or edited prose at roughly 100 characters. Existing documents may retain their historical formatting; reflow gradually when making substantive edits.
- Use [GitHub Flavored Markdown](https://github.github.com/gfm/) features as needed. Tables and code fences may exceed the wrap limit.
- Run `mdformat --wrap no` on any Markdown file you touch to ensure consistent spacing.

## Structure

- Store status updates, retrospectives, and experiment reports in `docs/reports/`.
- Store engineering-only instructions in `docs/internal/`.
- Reference other documents using relative paths (e.g. `[report](../reports/EXAMPLE.md)`). Update those links when moving files.

## Validation

- If documentation includes code samples, prefer using tested snippets from the repository. For longer examples, note whether they were executed.
- When documentation changes accompany code changes, ensure the tests described in the associated PR have been run.
