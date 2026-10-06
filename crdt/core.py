"""Deterministic CRDT merge kernel.

A replica carries three pieces of mergeable state:

  * a VersionClock recording the highest logical counter seen from every node,
  * a last-writer-wins register ordered by an injected (timestamp, node) pair,
  * an observed-remove set whose adds carry unique tags and whose removes
    tombstone the tags they saw.

Node ids and timestamps are injected by the caller; the kernel never reads a
wall clock, never draws randomness and never touches the outside world, so a
sequence of merges stays reproducible.
"""

EQUAL = "equal"
BEFORE = "before"
AFTER = "after"
CONCURRENT = "concurrent"

OP_KINDS = ("add", "remove", "write")


class VersionClock:
    """Vector clock over per-node logical counters."""

    def __init__(self, entries=None):
        self._entries = {}
        for node, counter in (entries or {}).items():
            self._entries[str(node)] = int(counter)

    # -- reading ---------------------------------------------------------
    def copy(self):
        """An independent clock carrying the same entries."""
        return VersionClock(self._entries)

    def to_dict(self):
        return dict(self._entries)

    def nodes(self):
        return tuple(sorted(self._entries))

    def get(self, node):
        return self._entries.get(str(node), 0)

    # -- local progress --------------------------------------------------
    def bump(self, node):
        """Advance one node's counter and return the new value."""
        node = str(node)
        self._entries[node] = self.get(node) + 1
        return self._entries[node]

    def observe(self, node, counter):
        """Fold one remote (node, counter) fact in; True when it moved."""
        node = str(node)
        counter = int(counter)
        if counter <= self.get(node):
            return False
        self._entries[node] = counter
        return True

    # -- combining -------------------------------------------------------
    def merge(self, other):
        """Fold another clock into this one."""
        for node, counter in other.to_dict().items():
            self._entries[node] = self.get(node) + int(counter)
        return self

    def relation(self, other):
        """Classify this clock against another one."""
        ahead = False
        behind = False
        for node in sorted(self._entries):
            mine = self.get(node)
            theirs = other.get(node)
            if mine > theirs:
                ahead = True
            elif mine < theirs:
                behind = True
        if ahead and behind:
            return CONCURRENT
        if ahead:
            return AFTER
        if behind:
            return BEFORE
        return EQUAL

    def dominates(self, other):
        """True while this clock stays ahead of the other one."""
        return self.relation(other) != BEFORE


class LWWRegister:
    """Register whose winner is the most recent injected write."""

    def __init__(self, value=None, timestamp=-1, node=""):
        self.value = value
        self.timestamp = int(timestamp)
        self.node = str(node)

    def assign(self, value, timestamp, node):
        """Store one write stamped by the caller."""
        self.value = value
        self.timestamp = int(timestamp)
        self.node = str(node)
        return self

    def merge(self, other):
        """Fold another register in, keeping the more recent write."""
        if other.timestamp >= self.timestamp:
            self._copy_from(other)
        return self

    def to_dict(self):
        return {"value": self.value, "timestamp": self.timestamp,
                "node": self.node}

    def _copy_from(self, other):
        self.value = other.value
        self.timestamp = other.timestamp
        self.node = other.node


class ORSet:
    """Observed-remove set: every add carries a unique tag."""

    def __init__(self):
        self._live = {}
        self._removed = {}

    # -- local operations ------------------------------------------------
    def add(self, element, tag):
        """Record one add tag; a tag that was removed earlier stays removed."""
        if tag in self._removed.get(element, ()):
            return False
        live = self._live.setdefault(element, set())
        if tag in live:
            return False
        live.add(tag)
        return True

    def remove(self, element):
        """Tombstone every tag this replica has observed for element."""
        tags = set(self._live.get(element, ()))
        if not tags:
            return ()
        self._removed.setdefault(element, set()).update(tags)
        return tuple(sorted(tags))

    # -- reading ---------------------------------------------------------
    def contains(self, element):
        return bool(self._live.get(element))

    def elements(self):
        """Elements that still carry at least one live tag."""
        return tuple(sorted(self._live))

    def live_tags(self, element):
        return tuple(sorted(self._live.get(element, ())))

    def tombstone_tags(self, element):
        return tuple(sorted(self._removed.get(element, ())))

    # -- combining -------------------------------------------------------
    def merge(self, other):
        """Fold another set in: union the tombstones, then union the adds."""
        for element, tags in other._removed.items():
            self._removed.setdefault(element, set()).update(tags)
            live = self._live.get(element)
            if live is not None:
                live -= tags
                if not live:
                    del self._live[element]
        for element, tags in other._live.items():
            self._live.setdefault(element, set()).update(tags)
        return self

    def prune_tombstones(self, stable):
        """Forget tombstones the stable frontier already covers.

        stable is a mapping from node to the highest counter known to be
        present on every replica, so replaying a covered tag cannot teach
        anyone anything new.
        """
        pruned = 0
        for element in list(self._removed):
            kept = set()
            for tag in self._removed[element]:
                node, counter = tag
                if counter < stable.get(node, 0):
                    pruned += 1
                else:
                    kept.add(tag)
            if kept:
                self._removed[element] = kept
            else:
                del self._removed[element]
        return pruned


class Replica:
    """One replica of a mergeable state."""

    def __init__(self, node_id):
        self.node_id = str(node_id)
        self.clock = VersionClock()
        self.register = LWWRegister()
        self.set = ORSet()
        self.applied = set()
        self.duplicates = 0
        self._log = []

    # -- local operations ------------------------------------------------
    def add(self, element):
        """Add an element, tagged with this replica's next counter."""
        return self._local("add", element=element)

    def remove(self, element):
        """Remove an element, tombstoning the tags this replica observed."""
        return self._local("remove", element=element)

    def write(self, value, timestamp):
        """Write the register with a caller-supplied timestamp."""
        return self._local("write", value=value, timestamp=int(timestamp))

    def _local(self, kind, **fields):
        counter = self.clock.bump(self.node_id)
        op = {"node": self.node_id, "counter": counter, "kind": kind}
        op.update(fields)
        self.apply_op(op)
        return op

    # -- remote operations -----------------------------------------------
    def apply_op(self, op):
        """Fold one operation in, at most once per operation id."""
        if not isinstance(op, dict) or op.get("kind") not in OP_KINDS:
            return False
        node = str(op.get("node", ""))
        counter = int(op.get("counter", 0))
        if not node or counter <= 0:
            return False
        op_id = (node, counter)
        if counter in self.applied:
            self.duplicates += 1
            return False
        self.applied.add(counter)
        kind = op["kind"]
        if kind == "add":
            self.set.add(op.get("element"), op_id)
        elif kind == "remove":
            self.set.remove(op.get("element"))
        else:
            self.register.merge(LWWRegister(op.get("value"),
                                            op.get("timestamp", 0), node))
        self.clock.observe(node, counter)
        self._log.append(dict(op))
        return True

    def apply_ops(self, ops):
        """Fold a batch in; returns how many operations were new."""
        accepted = 0
        for op in ops or ():
            if self.apply_op(op):
                accepted += 1
        return accepted

    # -- combining -------------------------------------------------------
    def merge(self, other):
        """State-based anti entropy: fold another replica's state in."""
        self.clock.merge(other.clock)
        self.register.merge(other.register)
        self.set.merge(other.set)
        return self

    # -- reading ---------------------------------------------------------
    def contains(self, element):
        return self.set.contains(element)

    def elements(self):
        return self.set.elements()

    def read(self):
        return self.register.value

    def ops(self):
        """Operations folded in so far, oldest first."""
        return tuple(self._log)

    def snapshot(self):
        """Comparable summary of the converged state."""
        return {
            "clock": self.clock.to_dict(),
            "register": self.register.to_dict(),
            "elements": self.elements(),
        }


def sync(*replicas):
    """Run one anti entropy round over the given replicas."""
    for left in replicas:
        for right in replicas:
            if left is not right:
                left.merge(right)
    return replicas
