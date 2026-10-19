from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


def build_brainscape_index(
    records: list[MutableMapping[str, Any]],
    *,
    match_group: str,
    match_keys: Sequence[str],
    modality_key: str,
) -> dict[tuple[str, ...], tuple[int, str]]:
    """Index records by strict meta keys + modality.

    Returns:
      idx[key_tuple] = (record_index, relpath_in_match_group)

    where:
      key_tuple = (dataset, subject, session, type, group, modality) by default.

    Notes:
      - `modality` is not necessarily a top-level record field; it is taken from the keys
        of `record[match_group]` (a dict modality -> relative path).
      - We keep the relative path around to validate that the batch path matches the JSON entry.
    """
    if modality_key not in match_keys:
        raise ValueError(
            "match_keys must include modality_key at the end (strict matching)."
        )

    idx: dict[tuple[str, ...], tuple[int, str]] = {}
    dups: list[tuple[str, ...]] = []

    for i, r in enumerate(records):
        group_paths = r.get(match_group)
        if not isinstance(group_paths, dict):
            raise ValueError(
                f"group:{match_group} must be present for each record entry."
            )

        # Base tuple for all non-modality keys.
        base = [str(r.get(k)) for k in match_keys if k != modality_key]

        for mod, rel in group_paths.items():
            if rel is None:
                raise ValueError(
                    f"Value is missing for modality key:{mod}, in record entry: {base}"
                )

            key = tuple(base + [str(mod)])
            if key in idx:
                dups.append(key)
            else:
                idx[key] = (i, str(rel))

    if dups:
        sample = "\n".join(str(k) for k in dups[:10])
        raise ValueError(
            f"Duplicate record keys detected for match_keys={tuple(match_keys)}.\n"
            f"First duplicates (up to 10):\n{sample}\n"
            "Fix by adding more match keys or resolving duplicates in the datalist."
        )
    return idx


CANONICAL_RECORD_KEY_FIELDS = (
    "dataset",
    "subject",
    "session",
    "type",
    "group",
    "modality",
)


def canonical_record_key(record: Mapping[str, Any], modality: str) -> tuple[str, ...]:
    """Return the stable BrainScape record identity for one modality."""
    values: list[str] = []
    for field in CANONICAL_RECORD_KEY_FIELDS[:-1]:
        if field not in record:
            raise KeyError(f"Record is missing canonical identity field {field!r}")
        values.append(str(record[field]))
    values.append(str(modality).lower())
    return tuple(values)


def canonical_record_identity(
    record: Mapping[str, Any], modality: str
) -> dict[str, str]:
    """Return the named canonical identity for one BrainScape record/modality."""
    return dict(
        zip(
            CANONICAL_RECORD_KEY_FIELDS,
            canonical_record_key(record, modality),
            strict=True,
        )
    )


def record_key_string(key: Sequence[str]) -> str:
    """Serialize a canonical key for a JSONL field without changing its order."""
    import json

    return json.dumps(list(key), separators=(",", ":"))


def resolve_record_file(
    record: Mapping[str, Any],
    *,
    root: Path,
    record_key: str,
    file_type: str,
    modality: str,
    group_dir: str,
) -> Path:
    """Resolve and validate a file referenced by a BrainScape manifest record.

    The expected manifest layout is ``record[record_key][file_type][modality]``;
    the referenced path is relative to ``root / dataset / group_dir``.
    """
    dataset = record.get("dataset")
    if dataset is None:
        raise KeyError("Record has no dataset field")
    try:
        file_map = record[record_key][file_type]
    except KeyError as exc:
        raise KeyError(
            f"Missing {record_key}.{file_type} for record {dataset!r}"
        ) from exc
    if not isinstance(file_map, Mapping):
        raise TypeError(
            f"{record_key}.{file_type} must be a mapping for record {dataset!r}"
        )
    matches = [key for key in file_map if str(key).lower() == str(modality).lower()]
    if len(matches) != 1:
        raise KeyError(
            f"Missing unambiguous {record_key}.{file_type}.{modality} for record {dataset!r}"
        )
    path = root / str(dataset) / group_dir / str(file_map[matches[0]])
    if not path.is_file():
        raise FileNotFoundError(f"Referenced {file_type} file is missing: {path}")
    return path


def _policy_values(value: object) -> tuple[str, ...]:
    """Treat a scalar or sequence policy field uniformly."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (str(value),)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


def brainscape_exclusion_reason(
    record: Mapping[str, Any], policy: Mapping[str, Any]
) -> str | None:
    """Return a configured dataset or pathology-field exclusion reason.

    ``subject_exclusions`` rules are intentionally flexible:
    ``dataset`` can be null, ``all``, one dataset, or a dataset list;
    ``demographic_field`` can be one field or a list; and ``target_values``
    can be one value or a list. ``true_values`` remains accepted for existing
    configurations during the transition.
    """
    dataset = str(record.get("dataset", ""))
    dataset_value = dataset.strip().casefold()
    excluded_datasets = {
        value.strip().casefold()
        for value in _policy_values(policy.get("excluded_datasets", ()))
    }
    if dataset_value in excluded_datasets:
        return f"dataset:{dataset}"

    demographics = record.get("demographics")
    if not isinstance(demographics, Mapping):
        return None
    demographic_keys = {str(key).strip().casefold(): str(key) for key in demographics}
    for rule_value in policy.get("subject_exclusions", ()):
        if not isinstance(rule_value, Mapping):
            continue
        datasets = {
            value.strip().casefold()
            for value in _policy_values(rule_value.get("dataset"))
            if value.strip()
        }
        if datasets and "all" not in datasets and dataset_value not in datasets:
            continue
        fields = _policy_values(rule_value.get("demographic_field"))
        target_values = _policy_values(
            rule_value.get("target_values", rule_value.get("true_values", ()))
        )
        targets = {value.strip().casefold() for value in target_values}
        if not targets:
            continue
        for configured_field in fields:
            field_key = configured_field.strip().casefold()
            actual_field = demographic_keys.get(field_key)
            if actual_field is None:
                continue
            if str(demographics[actual_field]).strip().casefold() in targets:
                return f"{dataset}:{actual_field}"
    return None
