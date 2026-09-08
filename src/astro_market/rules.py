"""Simple boolean DSL for combining atoms: & | ~ and parentheses."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


class RuleParseError(ValueError):
    """Invalid rule expression."""


_ATOM_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _normalize_expr(expr: str) -> str:
    """
    Convert DSL operators to Python AST-friendly form.
    DSL: atom names with & | ~ and parentheses.
    Map: & -> and, | -> or, ~ -> not  (for parsing only; eval uses pandas ops).
    """
    s = expr.strip()
    if not s:
        raise RuleParseError("Empty rule expression")

    tokens: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "()":
            tokens.append(c)
            i += 1
            continue
        if c == "&":
            tokens.append("and")
            i += 1
            continue
        if c == "|":
            tokens.append("or")
            i += 1
            continue
        if c == "~":
            tokens.append("not")
            i += 1
            continue
        if c.isalpha() or c == "_":
            j = i + 1
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            ident = s[i:j]
            if not _ATOM_RE.match(ident):
                raise RuleParseError(f"Invalid atom name: {ident!r}")
            tokens.append(ident)
            i = j
            continue
        raise RuleParseError(f"Unexpected character {c!r} in rule: {expr!r}")

    return " ".join(tokens)


@dataclass
class ParsedRule:
    """Parsed rule with original string, pythonized expr, and atom names used."""

    source: str
    py_expr: str
    atoms: list[str]
    tree: ast.Expression


class _AtomCollector(ast.NodeVisitor):
    """Validate AST and collect Name nodes (atom references)."""

    ALLOWED = (
        ast.Expression,
        ast.BoolOp,
        ast.UnaryOp,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Name,
        ast.Load,
    )

    def __init__(self) -> None:
        self.names: list[str] = []

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, self.ALLOWED):
            raise RuleParseError(
                f"Disallowed syntax in rule: {type(node).__name__}"
            )
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in ("and", "or", "not", "True", "False"):
            raise RuleParseError(f"Reserved name not allowed as atom: {node.id}")
        self.names.append(node.id)


def parse_rule(expr: str) -> ParsedRule:
    """Parse a DSL rule string into a validated AST."""
    py_expr = _normalize_expr(expr)
    try:
        tree = ast.parse(py_expr, mode="eval")
    except SyntaxError as e:
        raise RuleParseError(f"Syntax error in rule {expr!r}: {e}") from e

    collector = _AtomCollector()
    collector.visit(tree)
    seen: set[str] = set()
    atoms: list[str] = []
    for n in collector.names:
        if n not in seen:
            seen.add(n)
            atoms.append(n)

    return ParsedRule(source=expr.strip(), py_expr=py_expr, atoms=atoms, tree=tree)


def _eval_node(node: ast.AST, env: dict[str, pd.Series]) -> pd.Series:
    """Recursively evaluate boolean AST using pandas element-wise ops."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env)

    if isinstance(node, ast.Name):
        return env[node.id].astype(bool)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return ~_eval_node(node.operand, env).astype(bool)

    if isinstance(node, ast.BoolOp):
        values = [_eval_node(v, env).astype(bool) for v in node.values]
        out = values[0]
        if isinstance(node.op, ast.And):
            for v in values[1:]:
                out = out & v
        elif isinstance(node.op, ast.Or):
            for v in values[1:]:
                out = out | v
        else:
            raise RuleParseError(f"Unsupported BoolOp: {type(node.op).__name__}")
        return out.astype(bool)

    raise RuleParseError(f"Cannot evaluate node type: {type(node).__name__}")


def evaluate_ast(parsed: ParsedRule, atoms_df: pd.DataFrame) -> pd.Series:
    """
    Evaluate a parsed rule against a boolean atom DataFrame.
    Missing atoms raise KeyError with a clear message.
    """
    missing = [a for a in parsed.atoms if a not in atoms_df.columns]
    if missing:
        raise KeyError(f"Atoms not found in frame: {missing}")

    env = {a: atoms_df[a] for a in parsed.atoms}
    result = _eval_node(parsed.tree, env)
    out = result.astype(bool)
    out.name = parsed.source
    return out


def evaluate_rule_signal(expr: str, atoms_df: pd.DataFrame) -> pd.Series:
    """Parse + evaluate convenience wrapper."""
    return evaluate_ast(parse_rule(expr), atoms_df)


def rule_complexity(expr: str) -> int:
    """
    Complexity = number of atom references + number of operators.
    Used as a fitness penalty term.
    """
    parsed = parse_rule(expr)
    n_ops = 0

    class OpCounter(ast.NodeVisitor):
        def visit_BoolOp(self, node: ast.BoolOp) -> Any:
            nonlocal n_ops
            n_ops += max(0, len(node.values) - 1)
            self.generic_visit(node)

        def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
            nonlocal n_ops
            n_ops += 1
            self.generic_visit(node)

    OpCounter().visit(parsed.tree)
    return len(parsed.atoms) + n_ops
