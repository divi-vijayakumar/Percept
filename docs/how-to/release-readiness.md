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
      and no files outside the documented package surface.

## 3. Artifact content

File names alone do not prove that published content is appropriate. Docstrings become API
documentation, and contract docstrings also feed generated JSON Schema descriptions. Audit source,
generated files, and built artifacts before upload.

- [ ] The public-content policy passes over shipped source and the repository's public documentation,
      tests, examples, and evaluation tooling:
      ```bash
      cd packages/percept-vision
      python3 -m pytest -q tests/test_no_card_ids.py
      ```
- [ ] Every work-item or section citation in published content resolves to a tracked public document;
      otherwise, replace it with self-contained technical prose.
- [ ] `docs/adr/` is a tracked, public part of the repository. Keep ADR citations when they identify a
      real public ADR, and preserve the decision's technical rationale.
- [ ] The same policy passes against the unpacked sdist and wheel contents. Generated schemas count as
      published API content, not build by-products.
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
