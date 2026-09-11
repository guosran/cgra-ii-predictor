#!/usr/bin/env python3
"""Extract one standalone pre-mapper Neura DFG module per Taskflow task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

from amoeba_protocol import SOURCE_TASK_BODY_SHA_ATTR


def _balanced_region(lines: Sequence[str], start: int) -> Tuple[List[str], int]:
    depth = 0
    opened = False
    result = []
    for index in range(start, len(lines)):
        line = lines[index]
        result.append(line)
        depth += line.count("{") - line.count("}")
        opened = opened or "{" in line
        if opened and depth == 0:
            return result, index + 1
    raise ValueError("unterminated MLIR region")


def _split_types(value: str) -> List[str]:
    result = []
    start = 0
    depth = 0
    for index, character in enumerate(value):
        if character in "<([{" :
            depth += 1
        elif character in ">)]}" :
            depth -= 1
        elif character == "," and depth == 0:
            result.append(value[start:index].strip())
            start = index + 1
    result.append(value[start:].strip())
    if any(not item for item in result):
        raise ValueError("kernel input type list is invalid")
    return result


def _parse_operand_group(header: str, keyword: str) -> Tuple[List[str], List[str]]:
    match = re.search(
        rf"\b{re.escape(keyword)}\((.*?)\s+:\s+(.*?)\)",
        header,
    )
    if match is None:
        return [], []
    operands = [item.strip() for item in match.group(1).split(",")]
    types = _split_types(match.group(2))
    if len(operands) != len(types) or any(
        re.fullmatch(r"%[A-Za-z0-9_.$-]+", operand) is None
        for operand in operands
    ):
        raise ValueError(f"kernel {keyword} operands/types differ")
    return operands, types


def _normalize_empty_store_indexed(line: str) -> str:
    """Use generic syntax for the unparsable zero-rank custom form.

    Neura currently prints a base-less, zero-index store as
    ``neura.store_indexed %v to [ : ] ...``.  The custom parser rejects that
    spelling because ``type($indices)`` cannot parse an empty type list.  The
    equivalent generic syntax is round-trip parseable and keeps all three
    operand segment sizes explicit.
    """

    match = re.fullmatch(
        r"(\s*)neura\.store_indexed\s+(%[A-Za-z0-9_.$-]+)\s+to\s+"
        r"\[\s*:\s*\]\s*(\{.*\})?\s*:\s*(.+?)\s*",
        line,
    )
    if match is None:
        return line
    indentation, value, attributes, value_type = match.groups()
    attribute_text = f" {attributes}" if attributes else ""
    return (
        f'{indentation}"neura.store_indexed"({value}) '
        "<{operandSegmentSizes = array<i32: 1, 0, 0>}>"
        f"{attribute_text} : ({value_type}) -> ()"
    )


def _source_task_body_sha256(task_name: str,
                             task_region: Sequence[str]) -> str:
    """Read the exact body identity written by Amoeba during enumeration."""
    matches = re.findall(
        rf'\b{re.escape(SOURCE_TASK_BODY_SHA_ATTR)}\s*=\s*"([0-9a-f]{{64}})"',
        "\n".join(task_region),
    )
    if len(matches) != 1:
        raise ValueError(
            f"task {task_name} needs exactly one "
            f"{SOURCE_TASK_BODY_SHA_ATTR} attribute; run Amoeba candidate "
            "enumeration and extract DFGs from its bound IR output"
        )
    return matches[0]


def _select_function_lines(lines: Sequence[str], function: str) -> List[str]:
    """Return one requested ``func.func`` region from an MLIR module."""
    matches = [
        index for index, line in enumerate(lines)
        if re.search(
            rf'\bfunc\.func\s+@(?:{re.escape(function)}|'
            rf'"{re.escape(function)}")'
            r"(?![A-Za-z0-9_.$-])",
            line,
        )
    ]
    if not matches:
        raise ValueError(f"requested function {function} does not exist")
    if len(matches) != 1:
        raise ValueError(f"requested function {function} is ambiguous")
    selected, _ = _balanced_region(lines, matches[0])
    return selected


def extract_task_dfg_texts(
    text: str, function: Optional[str] = None,
) -> Dict[str, str]:
    lines = text.splitlines()
    if function is not None:
        if not function:
            raise ValueError("requested function name is empty")
        lines = _select_function_lines(lines, function)
    result: Dict[str, str] = {}
    index = 0
    while index < len(lines):
        task_match = re.search(r"\btaskflow\.task\s+@([A-Za-z0-9_.$-]+)", lines[index])
        if task_match is None:
            index += 1
            continue
        task_name = task_match.group(1)
        task_region, index = _balanced_region(lines, index)
        source_body_sha = _source_task_body_sha256(task_name, task_region)
        kernel_starts = [
            position for position, line in enumerate(task_region)
            if re.search(r"\bneura\.kernel\b", line)
        ]
        if not kernel_starts:
            continue
        if len(kernel_starts) != 1:
            raise ValueError(f"task {task_name} must contain exactly one Neura kernel")
        kernel_lines, _ = _balanced_region(task_region, kernel_starts[0])
        header = kernel_lines[0]
        input_operands, input_types = _parse_operand_group(header, "inputs")
        iter_operands, iter_types = _parse_operand_group(header, "iter_args_init")
        operands = input_operands + iter_operands
        types = input_types + iter_types
        if not operands:
            raise ValueError(f"task {task_name} kernel operand syntax is unsupported")
        standalone_operands = [f"%input{number}" for number in range(len(types))]
        replacements = dict(zip(operands, standalone_operands))
        kernel_lines[0] = re.sub(
            r"%[A-Za-z0-9_.$-]+",
            lambda match: replacements.get(match.group(0), match.group(0)),
            header,
        )
        kernel_lines = [_normalize_empty_store_indexed(line) for line in kernel_lines]
        indentation = len(kernel_lines[0]) - len(kernel_lines[0].lstrip())
        dedented = [
            "    " + line[min(indentation, len(line) - len(line.lstrip())):]
            for line in kernel_lines
        ]
        function_args = ", ".join(
            f"{operand}: {kind}"
            for operand, kind in zip(standalone_operands, types)
        )
        symbol = re.sub(r"[^A-Za-z0-9_.$-]", "_", task_name)
        result[task_name] = "\n".join([
            "module attributes {",
            f'  {SOURCE_TASK_BODY_SHA_ATTR} = "{source_body_sha}"',
            "} {",
            f"  func.func @{symbol}_dfg({function_args}) {{",
            *dedented,
            "    return",
            "  }",
            "}",
            "",
        ])
    if not result:
        raise ValueError("input contains no Taskflow task with a Neura kernel")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-output", type=Path)
    parser.add_argument(
        "--function",
        help="Taskflow function to select before extracting its task DFGs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = extract_task_dfg_texts(args.input.read_text(), args.function)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for task, text in sorted(outputs.items()):
        path = (args.output_dir / f"{task}.mlir").resolve()
        path.write_text(text)
        index[task] = str(path)
    if args.index_output is not None:
        args.index_output.parent.mkdir(parents=True, exist_ok=True)
        args.index_output.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
