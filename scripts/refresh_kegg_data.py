"""Refresh the KEGG module, KO, and pathway mapping files from the KEGG REST API.

Existing files are backed up to ``*.bak`` before being overwritten.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
import time
from collections.abc import Iterable
from pathlib import Path

try:
    import requests
    from requests import Response

    _USE_REQUESTS = True
except ImportError:  # pragma: no cover - fallback for minimal environments
    import urllib.error
    import urllib.request

    _USE_REQUESTS = False

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - graceful degradation
    def tqdm(iterable: Iterable, **_: object) -> Iterable:  # type: ignore[misc]
        return iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
MAPPING_DIR = REPO_ROOT / "data" / "external" / "mapping"
MODULE_TSV = MAPPING_DIR / "module-definitions.tsv"
KO_JSON = MAPPING_DIR / "KO_dictionary.json"
PATHWAY_TSV = MAPPING_DIR / "pathway-ko-membership.tsv"

KEGG_BASE = "https://rest.kegg.jp"
REQUEST_DELAY_S = 0.34  # ~3 requests/second per KEGG guidance
HTTP_TIMEOUT_S = 30.0


def kegg_get(path: str) -> str:
    """Fetch a KEGG REST endpoint and return the response body as text.

    Parameters
    ----------
    path : str
        Path component appended to ``https://rest.kegg.jp`` (e.g. ``"/list/module"``).

    Returns
    -------
    str
        Decoded response body.

    Raises
    ------
    RuntimeError
        If the HTTP call fails or returns a non-200 status.
    """
    url = f"{KEGG_BASE}{path}"
    if _USE_REQUESTS:
        resp: Response = requests.get(url, timeout=HTTP_TIMEOUT_S)
        if resp.status_code != 200:
            raise RuntimeError(f"KEGG GET {url} -> HTTP {resp.status_code}")
        return resp.text
    try:  # pragma: no cover - urllib fallback
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as fh:
            return fh.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"KEGG GET {url} -> HTTP {exc.code}") from exc


def fetch_module_ids() -> list[str]:
    """Return the sorted list of KEGG module IDs.

    Returns
    -------
    list[str]
        Module IDs such as ``"M00001"``.
    """
    body = kegg_get("/list/module")
    ids: list[str] = []
    for line in body.splitlines():
        if not line.strip():
            continue
        mid = line.split("\t", 1)[0].strip()
        if mid:
            ids.append(mid)
    return sorted(set(ids))


def parse_module_record(text: str) -> tuple[str, list[str]]:
    """Parse a KEGG flat-file module record.

    Parameters
    ----------
    text : str
        Raw body returned by ``/get/Mxxxxx``.

    Returns
    -------
    tuple[str, list[str]]
        ``(name, definition_lines)`` where ``definition_lines`` is the list of
        non-empty lines comprising the DEFINITION block (one per alternative).
        ``name`` is empty if not found; ``definition_lines`` is empty if the
        DEFINITION section is absent.
    """
    name = ""
    definition_lines: list[str] = []
    section: str | None = None
    for raw in text.splitlines():
        if not raw:
            continue
        if raw[0] != " ":  # New section header
            head = raw[:12].rstrip()
            rest = raw[12:].rstrip()
            section = head
            if head == "NAME":
                name = rest.strip()
            elif head == "DEFINITION":
                if rest.strip():
                    definition_lines.append(rest.strip())
            continue
        # Continuation line for the current section.
        if section == "DEFINITION":
            cont = raw[12:].rstrip() if len(raw) > 12 else raw.strip()
            if cont.strip():
                definition_lines.append(cont.strip())
        elif section == "NAME" and not name:
            name = raw.strip()
    return name, definition_lines


def fetch_modules(module_ids: list[str]) -> tuple[list[tuple[str, int, str, str]], list[str]]:
    """Fetch each module's flat-file record and return TSV rows + failures.

    Parameters
    ----------
    module_ids : list[str]
        Module IDs to fetch.

    Returns
    -------
    tuple[list[tuple[str, int, str, str]], list[str]]
        Rows of ``(module_id, alternative_index, name, definition_line)`` and a
        list of module IDs that failed to fetch or parse.
    """
    rows: list[tuple[str, int, str, str]] = []
    failures: list[str] = []
    for mid in tqdm(module_ids, desc="Fetching modules", unit="mod"):
        try:
            body = kegg_get(f"/get/{mid}")
        except RuntimeError as exc:
            print(f"  WARN: {mid} fetch failed: {exc}", file=sys.stderr)
            failures.append(mid)
            time.sleep(REQUEST_DELAY_S)
            continue
        name, defn_lines = parse_module_record(body)
        if not defn_lines:
            print(f"  WARN: {mid} has no DEFINITION; skipping", file=sys.stderr)
            failures.append(mid)
        else:
            for idx, line in enumerate(defn_lines):
                rows.append((mid, idx, name, line))
        time.sleep(REQUEST_DELAY_S)
    return rows, failures


def fetch_ko_entries() -> dict[str, dict[str, str]]:
    """Fetch the full KO list and build the ``term_hash`` mapping.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping of ``"KO:Kxxxxx"`` to ``{"id": "Kxxxxx", "name": "..."}``.
    """
    body = kegg_get("/list/ko")
    term_hash: dict[str, dict[str, str]] = {}
    for line in body.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        ko_id, name = parts[0].strip(), parts[1].strip()
        if not ko_id.startswith("K"):
            continue
        term_hash[f"KO:{ko_id}"] = {"id": ko_id, "name": name}
    return dict(sorted(term_hash.items()))


def fetch_pathway_names() -> dict[str, str]:
    """Fetch the KEGG reference pathway-map catalog.

    Only ``map``-prefixed (organism-agnostic) reference maps are retained.

    Returns
    -------
    dict[str, str]
        Mapping of pathway ID (``"map00010"``) to display name.
    """
    body = kegg_get("/list/pathway")
    names: dict[str, str] = {}
    for line in body.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        pid, name = parts[0].strip(), parts[1].strip()
        if pid.startswith("map") and pid not in names:
            names[pid] = name
    return names


def fetch_pathway_ko_membership() -> dict[str, set[str]]:
    """Fetch KEGG pathway-map -> KO membership in a single API call.

    Restricted to reference ``path:mapXXXXX`` pathways, excluding the
    ``path:koXXXXX`` mirrors that carry identical content.

    Returns
    -------
    dict[str, set[str]]
        Mapping of pathway ID (``"map00010"``) to its set of KO members
        (``{"K00001", "K00002", ...}``).
    """
    body = kegg_get("/link/ko/pathway")
    members: dict[str, set[str]] = {}
    for line in body.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        pid_raw, ko_raw = parts[0].strip(), parts[1].strip()
        pid = pid_raw.removeprefix("path:")
        ko = ko_raw.removeprefix("ko:")
        if not pid.startswith("map") or not ko.startswith("K"):
            continue
        members.setdefault(pid, set()).add(ko)
    return members


def write_pathway_tsv(
    names: dict[str, str], members: dict[str, set[str]], path: Path
) -> None:
    """Write the pathway-KO membership TSV, sorted by pathway ID.

    Parameters
    ----------
    names : dict[str, str]
        Pathway ID -> display name (from :func:`fetch_pathway_names`).
    members : dict[str, set[str]]
        Pathway ID -> KO set (from :func:`fetch_pathway_ko_membership`).
    path : Path
        Output file path.
    """
    pathway_ids = sorted(set(names) | set(members))
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Pathway ID\tName\tKO IDs\n")
        for pid in pathway_ids:
            name = names.get(pid, "")
            ko_list = ",".join(sorted(members.get(pid, set())))
            fh.write(f"{pid}\t{name}\t{ko_list}\n")


def backup(path: Path) -> None:
    """Copy ``path`` to ``path.bak`` if the source exists.

    Parameters
    ----------
    path : Path
        File to back up.
    """
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        print(f"  backed up {path.name} -> {bak.name}")


def write_module_tsv(rows: list[tuple[str, int, str, str]], path: Path) -> None:
    """Write the module-definitions TSV in the canonical format.

    Parameters
    ----------
    rows : list[tuple[str, int, str, str]]
        ``(module_id, alternative, name, definition)`` tuples; written in the
        provided order after sorting by ``(module_id, alternative)``.
    path : Path
        Output file path.
    """
    rows_sorted = sorted(rows, key=lambda r: (r[0], r[1]))
    with path.open("w", encoding="utf-8") as fh:
        fh.write("Module ID\tAlternative\tName\tDefinition\n")
        for mid, alt, name, defn in rows_sorted:
            fh.write(f"{mid}\t{alt}\t{name}\t{defn}\n")


def write_ko_json(term_hash: dict[str, dict[str, str]], path: Path) -> None:
    """Write the KO dictionary JSON preserving the legacy schema.

    Parameters
    ----------
    term_hash : dict[str, dict[str, str]]
        Mapping of ``"KO:Kxxxxx"`` -> ``{"id", "name"}``.
    path : Path
        Output file path.
    """
    payload = {
        "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "format_version": "N/A",
        "data_version": "kegg-rest-list-ko",
        "ontology": "kegg_orthology",
        "term_hash": term_hash,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def main() -> int:
    """Run the refresh end-to-end.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    t0 = time.time()
    print(f"Refreshing KEGG data into {MAPPING_DIR}")

    # --- Modules ---------------------------------------------------------
    print("Fetching module list...")
    module_ids = fetch_module_ids()
    print(f"  got {len(module_ids)} module IDs")

    rows, failures = fetch_modules(module_ids)
    print(f"  parsed {len(rows)} rows across {len({r[0] for r in rows})} modules")
    if failures:
        print(f"  {len(failures)} modules failed: {failures}")

    backup(MODULE_TSV)
    write_module_tsv(rows, MODULE_TSV)
    print(f"  wrote {MODULE_TSV}")

    # --- KOs -------------------------------------------------------------
    print("Fetching KO list...")
    term_hash = fetch_ko_entries()
    print(f"  got {len(term_hash)} KO entries")

    backup(KO_JSON)
    write_ko_json(term_hash, KO_JSON)
    print(f"  wrote {KO_JSON}")

    # --- Pathway maps ----------------------------------------------------
    print("Fetching reference pathway map catalog...")
    pathway_names = fetch_pathway_names()
    print(f"  got {len(pathway_names)} pathway names")
    print("Fetching pathway -> KO membership (single /link call)...")
    pathway_members = fetch_pathway_ko_membership()
    print(f"  got KO members for {len(pathway_members)} pathways")

    backup(PATHWAY_TSV)
    write_pathway_tsv(pathway_names, pathway_members, PATHWAY_TSV)
    print(f"  wrote {PATHWAY_TSV}")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
