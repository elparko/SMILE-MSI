# Releasing SMILE MSI

A copy-pasteable checklist for cutting a release of **SMILE MSI** (Python package
`smile_msi`). Pushing a `vX.Y.Z` tag is the trigger that builds the standalone
Windows + macOS bundles and publishes them to a GitHub Release. Everything before
the tag is housekeeping so the version is consistent everywhere it appears.

The version lives in **one** place — `smile_msi/__init__.py` `__version__` — and
`pyproject.toml` reads it from there (`dynamic = ["version"]` + `[tool.hatch.version]`).
The in-app *Help → Check for updates* compares this string against the latest tag on
[`elparko/SMILE-MSI`](https://github.com/elparko/SMILE-MSI/releases),
so the tag you push **must** match `__version__`.

Throughout, replace `X.Y.Z` with the version you're releasing (e.g. `0.3.1`).

---

## 0. Pre-flight

- [ ] You're on `main` (or the release branch) with a clean tree:
      ```bash
      git switch main && git pull && git status
      ```
- [ ] Decide the version number (semver). The current `__version__` is the source
      of truth — check it:
      ```bash
      python -c "import smile_msi; print(smile_msi.__version__)"
      ```

## 1. Bump the version (single source of truth)

- [ ] Edit **`smile_msi/__init__.py`** and set `__version__ = "X.Y.Z"` (drop any
      `-dev` suffix). This is the *only* file that defines the version — `pyproject.toml`
      reads it via Hatch, so do **not** add a version to `pyproject.toml`.
- [ ] Confirm the bump took:
      ```bash
      python -c "import smile_msi; print(smile_msi.__version__)"   # → X.Y.Z
      ```

## 2. Update the changelog

- [ ] In **`CHANGELOG.md`**, rename the `## [Unreleased]` heading to
      `## [X.Y.Z] — <short title>` and add today's date.
- [ ] Skim the `Added` / `Changed` / `Fixed` lists — drop anything that didn't
      actually ship, tidy wording.
- [ ] Leave a fresh empty `## [Unreleased]` block at the top for the next cycle
      (with `### Added` / `### Changed` / `### Fixed` stubs).

## 3. Verify the macOS `.app` version

`scripts/make_macos_app.sh` reads `__version__` out of `smile_msi/__init__.py` and
expands it into the bundle's `Info.plist` (`CFBundleVersion` /
`CFBundleShortVersionString`), so the bundle cannot drift. Do **not** hardcode a
version there — rebuild and confirm instead.

- [ ] Rebuild the bundle and check the plist picked up the new version:
      ```bash
      ./scripts/make_macos_app.sh
      plutil -extract CFBundleShortVersionString raw "SMILE MSI.app/Contents/Info.plist"   # → X.Y.Z
      ```
- [ ] Sanity-check that no stale version strings remain anywhere in source/packaging:
      ```bash
      grep -rn "X\.Y\.Z" .   # should only match smile_msi/__init__.py and CHANGELOG.md
      ```
      (`smile_msi/provenance.py` falls back to `__version__` when the installed-package
      metadata lookup fails, so it needs no bump.)

## 4. Verify tests are green

The build workflow does **not** run the test suite — `ci.yml` does, on push/PR. Run
locally before tagging so a red build can't reach a release.

- [ ] Full suite (engine + GUI), headless:
      ```bash
      QT_QPA_PLATFORM=offscreen uv run pytest -q
      ```
      A clean run is fully green. (The GUI tests set `offscreen` themselves, but
      exporting it makes a local run match CI exactly.)
- [ ] Optionally mirror CI's split (core engine on 3.10–3.12 happens in `ci.yml`; you
      can't reproduce all interpreters locally, but a single green local run is enough
      to tag — CI will catch the rest on the merge commit).

## 5. Commit the release prep

- [ ] Commit the version + changelog + macOS-app changes together:
      ```bash
      git add smile_msi/__init__.py CHANGELOG.md scripts/make_macos_app.sh
      git commit -m "Release X.Y.Z"
      git push
      ```
- [ ] Wait for **CI** (`.github/workflows/ci.yml`) to go green on the pushed commit
      before tagging:
      ```bash
      gh run watch        # or: gh run list --workflow=ci.yml
      ```

## 6. Tag and push (this triggers the build)

`.github/workflows/build-app.yml` runs on any pushed `v*` tag. The tag is what
turns a build into a published Release, so it must match `__version__` exactly.

- [ ] Create and push the annotated tag:
      ```bash
      git tag -a vX.Y.Z -m "SMILE MSI vX.Y.Z"
      git push origin vX.Y.Z
      ```
      (Or `git push --tags`.) The leading `v` is required — the build matches
      `tags: ["v*"]` and the update check parses `vX.Y.Z`.

## 7. Confirm the build workflow publishes artifacts

`build-app.yml` runs PyInstaller on **both** `windows-latest` and `macos-latest`
(it can't cross-compile), smoke-tests each frozen bundle headless
(`SMILE_MSI_SELFTEST=1`, offscreen), zips it, and — because this is a tag push —
attaches the zip to a GitHub Release (created if absent, release notes auto-generated).

- [ ] Watch the build:
      ```bash
      gh run watch --workflow=build-app.yml
      ```
- [ ] Both matrix jobs (`SMILE-MSI-windows`, `SMILE-MSI-macos`) must pass — including
      the **"Smoke-test the frozen bundle"** step, which catches a missing hidden import
      before it reaches users.
- [ ] Confirm the Release exists with **both** zips attached:
      ```bash
      gh release view vX.Y.Z
      ```
      You should see `SMILE-MSI-windows.zip` and `SMILE-MSI-macos.zip`.
- [ ] (Optional) Download and open the macOS bundle once to confirm it launches:
      ```bash
      gh release download vX.Y.Z --pattern "SMILE-MSI-macos.zip"
      ```
- [ ] Edit the auto-generated release notes if you want them to lead with the
      `CHANGELOG.md` highlights instead of the raw commit list.

> A frozen bundle can't update itself, so the Release is the only way existing users
> get the new version — *Help → Check for updates* points them at this Release page
> when its tag is newer than their running `__version__`.

## 8. Open the next development cycle

- [ ] Bump `smile_msi/__init__.py` to the next dev version, e.g.
      `__version__ = "X.Y.(Z+1)-dev"`, so in-progress builds report ahead of the release:
      ```bash
      # edit __init__.py, then:
      git add smile_msi/__init__.py
      git commit -m "Back to development: X.Y.(Z+1)-dev"
      git push
      ```
      (The update-check version parser stops at the first non-numeric component, so a
      `-dev` suffix is safe and won't break the compare.)
- [ ] The `## [Unreleased]` block from step 2 is already in place for the next batch
      of changes.

---

## Reference — what runs when

| File | Trigger | Does |
| --- | --- | --- |
| `.github/workflows/ci.yml` | push to `main`, any PR | Engine tests on 3.10/3.11/3.12 + full GUI suite (offscreen) |
| `.github/workflows/build-app.yml` | push `v*` tag, or manual *Run workflow* | PyInstaller bundles for Windows + macOS, smoke-test, zip → Release (tag) or artifact (manual) |
| `.github/workflows/agent-fix.yml` | in-app feature requests (`agent-fix` label) | Unrelated to releases |
| `scripts/make_macos_app.sh` | run locally | Builds the dev `SMILE MSI.app` that wraps the repo venv (not the frozen release bundle); plist version is hardcoded — see step 3 |

A manual **Run workflow** on `build-app.yml` (no tag) builds the same bundles and
uploads them as downloadable artifacts **without** creating a Release — handy for a
dry run before tagging.
