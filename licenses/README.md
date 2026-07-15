# Bundled third-party license texts

These are the verbatim license texts SMILE MSI is obliged to distribute alongside its
**built binaries** (see [`../THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md) and
[`../NOTICE`](../NOTICE)):

| File | Why it's here |
|---|---|
| `LGPL-3.0.txt` | Qt for Python (**PySide6 / shiboken6**) is used under the LGPL-3.0. Distributing the frozen app requires shipping this text. |
| `GPL-3.0.txt` | The LGPL-3.0 is written as a set of additional permissions on top of the GPL-3.0 and incorporates it by reference, so the GPL-3.0 text ships too. |
| `CC-BY-4.0.txt` | The **Font Awesome** icon set (rendered via qtawesome) is licensed under CC BY 4.0 for the icons. |
| `Font-Awesome-CC-BY-4.0.txt` | The required Font Awesome attribution notice. |

`smile_msi.spec` copies this folder into the frozen bundle (`dist/SMILE MSI/licenses/`), so
every distributed `.exe` / `.app` ships with them next to the executable.

The canonical texts were retrieved from gnu.org (LGPL/GPL) and creativecommons.org (CC BY).
SMILE MSI itself is Apache-2.0 (see [`../LICENSE`](../LICENSE)); these files are the licenses
of *dependencies*, not of SMILE MSI.
