#!/usr/bin/env python3
"""
file_org_sim_clean.py – full simulation with all four file-organization methods,
including benchmarking and optional Tkinter GUI. This variant has been
cleaned: removed the progress-bar helper and non-essential comments, while
retaining the GUI and the _sizeof() memory estimator.

Run:
    python3 file_org_sim_clean.py [--gui]
"""

import argparse
import random
import string
import sys
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]

# ---------- helpers ----------------------------------------------------------

def _random_ascii(n: int = 180) -> str:
    """Return a string of `n` printable ASCII characters."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))


def _sizeof(obj) -> int:
    """
    Recursively estimate memory usage of pure-Python objects.
    This is a heuristic and not a perfect accounting of process memory,
    but it gives a comparable metric across the structures built here.
    """
    seen = set()

    def _inner(o):
        if id(o) in seen:
            return 0
        seen.add(id(o))
        size = sys.getsizeof(o)
        if isinstance(o, dict):
            size += sum(_inner(k) + _inner(v) for k, v in o.items())
        elif isinstance(o, (list, tuple, set)):
            size += sum(_inner(i) for i in o)
        return size

    return _inner(obj)


# ---------- base class -------------------------------------------------------
class FileOrg(ABC):
    """Abstract file-organisation method."""

    def __init__(self, records: List[Tuple[int, str]]):
        self.records = records
        self._build()

    @abstractmethod
    def _build(self):
        """Construct the in-memory structure."""
        pass

    @abstractmethod
    def search(self, key: int) -> Optional[str]:
        """Return payload if key exists, else None."""
        pass

    def benchmark(self, hits: List[int], misses: List[int]) -> Tuple[float, float]:
        """Return (avg_hit_ns, avg_miss_ns)."""
        # warm-up
        for _ in range(10):
            self.search(0)

        # successful lookups
        start = time.perf_counter_ns()
        for k in hits:
            self.search(k)
        hit_time = (time.perf_counter_ns() - start) / len(hits)

        # unsuccessful lookups
        start = time.perf_counter_ns()
        for k in misses:
            self.search(k)
        miss_time = (time.perf_counter_ns() - start) / len(misses)

        return hit_time, miss_time


# ---------- Sequential -------------------------------------------------------
class Sequential(FileOrg):
    """Linear scan through unsorted list."""

    def _build(self):
        # nothing to do – self.records already holds data
        pass

    def search(self, key: int) -> Optional[str]:
        for k, payload in self.records:
            if k == key:
                return payload
        return None

    def mem_size(self) -> int:
        return _sizeof(self.records)


# ---------- Indexed ----------------------------------------------------------
class Indexed(FileOrg):
    """Separate hash-map index: key -> block pointer (list index)."""

    def _build(self):
        self.index = {k: i for i, (k, _) in enumerate(self.records)}

    def search(self, key: int) -> Optional[str]:
        if key in self.index:
            return self.records[self.index[key]][1]
        return None

    def mem_size(self) -> int:
        return _sizeof(self.index) + _sizeof(self.records)


# ---------- Direct (Hashed) --------------------------------------------------
class Direct(FileOrg):
    """Hash table with chaining for collisions."""

    TABLE_SIZE = 200_003  # prime near 200k

    def _build(self):
        self.table: List[List[Tuple[int, str]]] = [[] for _ in range(self.TABLE_SIZE)]
        for k, payload in self.records:
            bucket = k % self.TABLE_SIZE
            self.table[bucket].append((k, payload))

    def search(self, key: int) -> Optional[str]:
        bucket = self.table[key % self.TABLE_SIZE]
        for k, payload in bucket:
            if k == key:
                return payload
        return None

    def mem_size(self) -> int:
        return _sizeof(self.table) + _sizeof(self.records)


# ---------- B-Tree (order 20) ------------------------------------------------
class BTreeNode:
    """Node for ORDER-order B-tree."""

    def __init__(self, leaf: bool = False):
        self.leaf = leaf
        self.keys: List[int] = []
        self.values: List[str] = []  # only used if leaf
        self.children: List["BTreeNode"] = []


class BTree(FileOrg):
    """ORDER-order B-tree: all data stored in leaves."""

    ORDER = 20
    MAX_KEYS = ORDER - 1

    def _build(self):
        self.root = BTreeNode(leaf=True)
        for k, payload in self.records:
            self._insert(k, payload)

    def _insert(self, key: int, payload: str):
        root = self.root
        if len(root.keys) == self.MAX_KEYS:
            new_root = BTreeNode(leaf=False)
            new_root.children.append(root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_nonfull(self.root, key, payload)

    def _insert_nonfull(self, node: BTreeNode, key: int, payload: str):
        if node.leaf:
            # insert into sorted order
            i = len(node.keys) - 1
            node.keys.append(0)
            node.values.append("")
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.values[i + 1] = node.values[i]
                i -= 1
            node.keys[i + 1] = key
            node.values[i + 1] = payload
        else:
            i = len(node.keys) - 1
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == self.MAX_KEYS:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_nonfull(node.children[i], key, payload)

    def _split_child(self, parent: BTreeNode, idx: int):
        full = parent.children[idx]
        new = BTreeNode(leaf=full.leaf)
        mid = self.MAX_KEYS // 2
        parent.keys.insert(idx, full.keys[mid])
        parent.children.insert(idx + 1, new)
        new.keys = full.keys[mid + 1:]
        new.values = full.values[mid + 1:]
        full.keys = full.keys[:mid]
        full.values = full.values[:mid]
        if not full.leaf:
            new.children = full.children[mid + 1:]
            full.children = full.children[:mid + 1]

    def search(self, key: int) -> Optional[str]:
        return self._search_node(self.root, key)

    def _search_node(self, node: BTreeNode, key: int) -> Optional[str]:
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and key == node.keys[i]:
            return node.values[i] if node.leaf else self._search_node(node.children[i], key)
        if node.leaf:
            return None
        return self._search_node(node.children[i], key)

    def mem_size(self) -> int:
        return _sizeof(self.root) + _sizeof(self.records)


# ---------- main driver ------------------------------------------------------
def run_simulation():
    """Run the benchmark and return results."""
    random.seed(42)
    RECORDS = 100_000
    LOOKUPS = 1_000

    print("Generating synthetic dataset…")
    keys = list(range(1_000_000))
    random.shuffle(keys)
    keys = keys[:RECORDS]
    records = [(k, _random_ascii()) for k in keys]

    # pick lookup keys
    hit_keys = random.sample([k for k, _ in records], LOOKUPS)
    key_set = set(keys)
    miss_keys = []
    while len(miss_keys) < LOOKUPS:
        candidate = random.randint(0, 1_000_000)
        if candidate not in key_set:
            miss_keys.append(candidate)

    methods = [
        ("Sequential", Sequential),
        ("Indexed", Indexed),
        ("Direct", Direct),
        ("B-Tree", BTree),
    ]

    results = []
    for name, cls in methods:
        print(f"\n=== {name} ===")
        inst = cls(records)
        hit_ns, miss_ns = inst.benchmark(hit_keys, miss_keys)
        mem = getattr(inst, "mem_size", lambda: _sizeof(inst))()
        results.append((name, hit_ns, miss_ns, mem / 1024 / 1024))

    return results


def print_results(results):
    """Print results to console."""
    print("\n" + "=" * 60)
    print(f"{'Method':<12}{'Avg Hit (ns)':>15}{'Avg Miss (ns)':>15}{'Memory (MB)':>12}")
    print("-" * 60)
    for name, h, m, mem in results:
        print(f"{name:<12}{h:>15.0f}{m:>15.0f}{mem:>12.1f}")

    max_time = max(h for _, h, _, _ in results)
    print("\nAccess-time spark-bar (hit):")
    for name, h, _, _ in results:
        bar_len = int(40 * h / max_time) if max_time > 0 else 0
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"{name:<12}|{bar}|")


# ---------- GUI --------------------------------------------------------------
def show_gui(results):
    """Display results in a Tkinter window."""
    root = tk.Tk()
    root.title("File Organization Benchmark Results")
    root.geometry("700x400")

    title_label = tk.Label(root, text="File Organization Performance", font=("Arial", 16, "bold"))
    title_label.pack(pady=10)

    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)

    tree = ttk.Treeview(frame, columns=("Method", "Avg Hit (ns)", "Avg Miss (ns)", "Memory (MB)"), show="headings", height=5)
    tree.heading("Method", text="Method")
    tree.heading("Avg Hit (ns)", text="Avg Hit (ns)")
    tree.heading("Avg Miss (ns)", text="Avg Miss (ns)")
    tree.heading("Memory (MB)", text="Memory (MB)")

    tree.column("Method", width=150, anchor="w")
    tree.column("Avg Hit (ns)", width=150, anchor="e")
    tree.column("Avg Miss (ns)", width=150, anchor="e")
    tree.column("Memory (MB)", width=150, anchor="e")

    for name, h, m, mem in results:
        tree.insert("", "end", values=(name, f"{h:.0f}", f"{m:.0f}", f"{mem:.1f}"))

    tree.pack(fill="both", expand=True)

    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    scrollbar.pack(side="right", fill="y")
    tree.configure(yscrollcommand=scrollbar.set)

    # Visual bar chart
    chart_frame = ttk.LabelFrame(root, text="Access Time Comparison (Hit)", padding=10)
    chart_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

    max_time = max(h for _, h, _, _ in results)
    for name, h, _, _ in results:
        row_frame = ttk.Frame(chart_frame)
        row_frame.pack(fill="x", pady=2)

        label = tk.Label(row_frame, text=f"{name:<12}", font=("Courier", 10), anchor="w", width=12)
        label.pack(side="left")

        bar_width = int(400 * h / max_time) if max_time > 0 else 0
        canvas = tk.Canvas(row_frame, width=400, height=20, bg="white", highlightthickness=0)
        canvas.pack(side="left", padx=5)
        canvas.create_rectangle(0, 0, bar_width, 20, fill="#4CAF50", outline="")

        time_label = tk.Label(row_frame, text=f"{h:.0f} ns", font=("Arial", 9))
        time_label.pack(side="left", padx=5)

    close_btn = ttk.Button(root, text="Close", command=root.destroy)
    close_btn.pack(pady=10)

    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="File Organization Simulation")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI and show results in console only")
    args = parser.parse_args()

    results = run_simulation()
    print_results(results)

    if not args.no_gui:
        if tk is None or ttk is None:
            print("\nError: Tkinter is not available. Cannot launch GUI.")
            sys.exit(1)
        show_gui(results)


if __name__ == "__main__":
    main()
