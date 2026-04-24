from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Any, Set

import numpy as np


def _binary_directions_from_raw(raw: np.ndarray) -> List[int]:
    """Map raw direction values to strictly ``1`` or ``-1`` (zeros -> ``1``)."""
    v = np.asarray(raw, dtype=float).ravel()
    s = np.sign(v).astype(np.int64)
    s[s == 0] = 1
    return [int(x) for x in s]


@dataclass
class TrieNode:
    """
    Single node in the prefix tree.

    Children are keyed by packet direction (1 or -1).
    At each node we keep a list of (label, trace_idx) pairs for all
    traces that pass through this node (prefix match).
    """

    children: Dict[int, "TrieNode"] = field(default_factory=dict)
    items: List[Tuple[int, int]] = field(default_factory=list)


class RadixTrie:
    """
    Prefix tree over traces of 1/-1 packet directions.

    Supports two build modes:
    1) build(traces): each trace is 1D with directions and trace[-1] = label.
    2) build_from_traces_and_labels(traces, labels): traces are 2D [time, dir]
       or 1D directions; labels are separate. Stores (label, trace_idx) for
       retrieving full traces for distance computation.
    """

    def __init__(
        self,
        traces: Iterable[Any] | None = None,
        labels: Iterable[int] | None = None,
    ) -> None:
        self.root = TrieNode()
        if traces is not None:
            if labels is not None:
                self.build_from_traces_and_labels(traces, labels)
            else:
                self.build(traces)

    # ------------------------------------------------------------------
    # Building the trie
    # ------------------------------------------------------------------
    def build(self, traces: Iterable[Any]) -> None:
        """
        Build from traces where each trace has label in the last position.
        Each trace: 1D array [d_0, ..., d_{T-1}, label].
        """
        for tr in traces:
            arr = np.asarray(tr)
            if arr.size < 2:
                continue
            if arr.ndim == 2:
                directions = _binary_directions_from_raw(arr[:-1, 1])
                label = int(arr[-1, 1])
            else:
                label = int(arr[-1])
                directions = _binary_directions_from_raw(arr[:-1])
            self.trace_insert(directions, label, -1)

    def build_from_traces_and_labels(
        self, traces: Iterable[Any], labels: Iterable[int]
    ) -> None:
        """
        Build from (traces, labels) and store (label, trace_idx) at each node.
        """
        traces_list = list(traces)
        labels_list = list(labels)
        if len(traces_list) != len(labels_list):
            raise ValueError("traces and labels must have the same length")
        for idx, (tr, label) in enumerate(zip(traces_list, labels_list)):
            arr = np.asarray(tr)
            if arr.size < 2:
                continue
            if arr.ndim == 2:
                directions = _binary_directions_from_raw(arr[:, 1])
            else:
                directions = [int(x) for x in arr.ravel()]
            self.trace_insert(directions, int(label), idx)

    def trace_insert(
        self, directions: Iterable[int], label: int, trace_idx: int = -1
    ) -> None:
        """Insert one labeled trace; trace_idx >= 0 for (label, trace_idx) storage."""
        node = self.root
        for d in directions:
            d = int(d)
            child = node.children.get(d)
            if child is None:
                child = TrieNode()
                node.children[d] = child
            node = child
        node.items.append((label, trace_idx))

    # ------------------------------------------------------------------
    # Search operations
    # ------------------------------------------------------------------
    def trace_find_node(self, prefix: Iterable[int]) -> TrieNode | None:
        """
        Traverse the trie according to the given prefix.

        Returns:
            The node reached by the prefix, or None if the prefix
            does not exist in the tree.
        """
        node = self.root
        for d in prefix:
            node = node.children.get(int(d))
            if node is None:
                return None
        return node

    def search(self, prefix: Iterable[int]) -> bool:
        """
        Check whether there exists at least one stored trace that has
        ``prefix`` as a prefix.

        Args:
            prefix: Iterable of 1/-1 directions (no label / index).

        Returns:
            True if the prefix exists in the tree, False otherwise.
        """
        return self.trace_find_node(prefix) is not None

    def collect_items(self, node: TrieNode, out: List[Tuple[int, int]]) -> None:
        """
        Collect all (label, trace_idx) pairs in the subtree rooted at node.
        """
        stack: List[TrieNode] = [node]
        while stack:
            cur = stack.pop()
            out.extend(cur.items)
            for child in cur.children.values():
                stack.append(child)

    def annotate_unique_trace_subtree_sizes(self) -> None:
        """
        For each node, set ``unique_trace_subtree_size`` to the number of
        distinct non-negative ``trace_idx`` values in that node's subtree.

        Enables O(prefix length) queries by walking from the root instead of
        calling :meth:`trace_match`, which scans the whole subtree each time.
        """
        if getattr(self.root, "_unique_subtree_annotated", False):
            return

        def dfs(node: TrieNode) -> Set[int]:
            acc: Set[int] = set()
            for _, tid in node.items:
                t = int(tid)
                if t >= 0:
                    acc.add(t)
            for ch in node.children.values():
                acc.update(dfs(ch))
            node.unique_trace_subtree_size = len(acc)
            return acc

        dfs(self.root)
        self.root._unique_subtree_annotated = True  # type: ignore[attr-defined]

    def unique_trace_match_count(self, prefix: Iterable[int]) -> int:
        """
        Number of distinct pool traces whose stored directions have ``prefix``
        as a prefix. Requires :meth:`annotate_unique_trace_subtree_sizes` once
        after building the trie (or uses subtree walk fallback if missing).
        """
        node = self.trace_find_node(prefix)
        if node is None:
            return 0
        size = getattr(node, "unique_trace_subtree_size", None)
        if size is not None:
            return int(size)
        pairs = self.trace_match(prefix)
        return len({int(tid) for _, tid in pairs if int(tid) >= 0})

    def trace_match(self, trace: Iterable[int]) -> List[Tuple[int, int]]:
        """
        Match a (possibly partial) trace against the prefix tree.

        Returns:
            List of (label, trace_idx) for all stored traces whose directions
            have ``trace`` as a prefix. trace_idx >= 0 when built with
            build_from_traces_and_labels; use it to index into the trace pool.
        """
        node = self.trace_find_node(trace)
        if node is None:
            return []

        matches: List[Tuple[int, int]] = []
        self.collect_items(node, matches)
        return matches

    def trace_match_min_idx_by_label(
        self, trace: Iterable[int]
    ) -> Dict[int, int]:
        """
        Like :meth:`trace_match`, but instead of returning *all* (label, trace_idx)
        pairs, returns only the minimum ``trace_idx`` per ``label``.

        This is useful when callers only need per-class minima and would
        otherwise allocate/iterate over a very large match list.
        """
        node = self.trace_find_node(trace)
        if node is None:
            return {}

        best: Dict[int, int] = {}
        stack: List[TrieNode] = [node]
        while stack:
            cur = stack.pop()
            for label, trace_idx in cur.items:
                lbl = int(label)
                idx = int(trace_idx)
                prev = best.get(lbl)
                if prev is None or idx < prev:
                    best[lbl] = idx
            for child in cur.children.values():
                stack.append(child)
        return best

