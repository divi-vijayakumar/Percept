# Release readiness

The pre-publish checklist for `percept-vision`. Run every item against the **built artifact**, not the
working tree — the two differ, and only the artifact is what the world gets.

Build first, then audit what you built:

```bash
cd packages/percept-vision
python3 -m build
```

## 1. Secrets

- [ ] `.env` is gitignored and has never been committed (`git log --all -- .env` is empty).
- [ ] No real keys in tracked files. Placeholders (`sk-or-...`) in docs are fine; run a real-key regex
      over the artifact, not just the repo:
      ```bash
      tar tzf dist/*.tar.gz >/dev/null && mkdir -p /tmp/rc && tar xzf dist/*.tar.gz -C /tmp/rc
      grep -rInE '(sk-[a-zA-Z0-9_-]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|xox[baprs]-|AIza[0-9A-Za-z_-]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY|eyJ[A-Za-z0-9_-]{20,}\.)' /tmp/rc
      ```

## 2. Artifact file manifest

- [ ] Top-level of the sdist is only: `LICENSE`, `README.md`, `PKG-INFO`, `pyproject.toml`,
      `setup.cfg`, `percept/`, `tests/`, `*.egg-info/`.
- [ ] No `.env`, no `.pem`/`.key`, no datasets (Charades / Ego4D), no `.mp4`, no eval fixtures,
      no internal planning docs.

## 3. Artifact **content** — the leak class a file manifest cannot catch

> **Learned the hard way on a pre-1.0 release.** That audit checked the sdist *file list*, found it
> clean (276 files, no keys, no datasets), and passed. It was looking for files that shouldn't ship. The
> actual leak was inside files that **should** ship: commercial-strategy language written into source
> docstrings — "the raw material of the **moat**", "so the **paid side** can price escalations",
> "`good: bool` … (False -> **billing credit on the paid side**)". A file-manifest check is
> structurally incapable of catching this, because every file involved was supposed to be there.
>
> It was worse than a stray comment for two reasons. First, `percept/contracts/` docstrings are the
> **source** of the generated JSON schemas, so one class docstring propagated the phrase "the OSS
> flywheel's keystone" into **26 published schema files** as a user-facing `description` — API
> documentation, not an internal note. Generated artifacts amplify a content leak; check the generated
> output too, not only the source it came from. Second, it shipped to PyPI, where releases are
> effectively permanent — mirrors and indexes retain the text, so yanking would not have retracted it.
>
> Scrub content before the upload, because after it there is no fix, only a follow-up release.

- [ ] No commercial/strategy language in the artifact:
      ```bash
      grep -rInE '(moat|flywheel|paid side|paid tier|OSS side|monetiz|billing credit|price escalation|upsell|free tier|enterprise tier|go-to-market|GTM|competitor|ARR|runway|TAM)' /tmp/rc
      ```
- [ ] No dangling citations to non-public documents (private specs, plans, ticket IDs, board cards).
      Every `§`/ticket reference in shipped source should resolve to something a reader can actually
      open, or be rewritten as self-describing prose:
      ```bash
      grep -rInE '(Viability §|spec §|impl plan §|PCPT-[0-9]+|board card|direction doc)' /tmp/rc
      ```
- [ ] The ADR set is internal (`internal-docs/adr/`, gitignored), so any `ADR-NNN` cited in shipped
      source or docs is **self-describing prose** (name + rationale) or points to public code — never a
      link to a non-public `docs/adr/` path.
- [ ] Generated schemas regenerate clean and match the source:
      ```bash
      python3 -m percept.contracts.jsonschema && git diff --exit-code -- percept/contracts/schema
      ```
- [ ] No local paths, hostnames, or personal identifiers beyond the intentional maintainer contact in
      `pyproject.toml`.

## 4. Install + runtime

- [ ] Wheel and sdist build cleanly.
- [ ] Clean-venv install works and `run()` fires end-to-end from the installed wheel.
- [ ] Console entry point (`percept-watch --goal "…"`) works from that clean venv.
- [ ] Full test suite green.

## 5. Licence, dependencies, egress

- [ ] Apache-2.0 present at repo root and in the package, declared in `pyproject.toml`.
- [ ] `pip-audit` clean for the runtime dependencies (findings in `pip` itself are user tooling, not
      the package).
- [ ] Webhook sinks block private/loopback and require https by default (`UrlPolicy`); `.dev()` is the
      explicit opt-in for local http.
- [ ] Frames egress only to the configured judge provider.

## 6. Policy documents

- [ ] `SECURITY.md` private-disclosure policy present, with the supported version range bumped to the
      release being cut.
- [ ] `CONTRIBUTING.md` present.
- [ ] README stability statement is honest about what is and isn't built.

## 7. Publish

- [ ] Version bumped in `pyproject.toml` and `CHANGELOG.md` entry written.
- [ ] Tag the release commit — a prior release shipped untagged, so there was no commit that reproduced it.
- [ ] `python3 -m twine upload dist/*`.
