"""Acceptance tests for the CRDT merge kernel."""

import unittest

from crdt.core import (AFTER, BEFORE, CONCURRENT, EQUAL, LWWRegister,
                       ORSet, Replica, VersionClock, sync)


class ClockRelationTests(unittest.TestCase):

    def test_relation_separates_concurrent_from_causal(self):
        base = VersionClock({"a": 2, "b": 1})
        self.assertEqual(base.relation(base.copy()), EQUAL)
        self.assertEqual(base.relation(VersionClock({"a": 3, "b": 1})), BEFORE)
        self.assertEqual(base.relation(VersionClock({"a": 2})), AFTER)
        self.assertEqual(VersionClock({"a": 2}).relation(base), BEFORE,
                         "a clock missing an entry is not concurrent")
        self.assertEqual(base.relation(VersionClock({"a": 1, "b": 2})),
                         CONCURRENT)
        self.assertEqual(VersionClock().relation(base), BEFORE)
        self.assertEqual(VersionClock({"c": 1}).relation(VersionClock({"a": 1})),
                         CONCURRENT)

    def test_dominance_requires_a_strictly_later_clock(self):
        base = VersionClock({"a": 2, "b": 1})
        later = VersionClock({"a": 3, "b": 1})
        other = VersionClock({"a": 1, "b": 5})
        self.assertTrue(later.dominates(base))
        self.assertFalse(base.dominates(base.copy()),
                         "a clock does not dominate itself")
        self.assertFalse(base.dominates(other))
        self.assertFalse(other.dominates(base))

    def test_merge_is_idempotent_and_takes_pointwise_maximum(self):
        clock = VersionClock({"a": 2})
        incoming = VersionClock({"a": 1, "b": 4})
        clock.merge(incoming)
        self.assertEqual(clock.to_dict(), {"a": 2, "b": 4})
        clock.merge(incoming)
        self.assertEqual(clock.to_dict(), {"a": 2, "b": 4})
        self.assertEqual(clock.relation(VersionClock({"a": 2, "b": 4})), EQUAL)


class RegisterTests(unittest.TestCase):

    def test_register_keeps_the_deterministic_winner(self):
        raised = LWWRegister("alpha", 5, "a")
        raised.merge(LWWRegister("beta", 5, "b"))
        other = LWWRegister("beta", 5, "b")
        other.merge(LWWRegister("alpha", 5, "a"))
        self.assertEqual(raised.value, other.value, "merge is not symmetric")
        self.assertEqual(raised.value, "beta")
        self.assertEqual(raised.node, "b")
        raised.merge(LWWRegister("gamma", 6, "a"))
        self.assertEqual(raised.value, "gamma")
        raised.merge(LWWRegister("delta", 4, "z"))
        self.assertEqual(raised.value, "gamma")


class ConvergenceTests(unittest.TestCase):

    def test_synced_replicas_agree_on_the_same_state(self):
        left = Replica("a")
        right = Replica("b")
        left.add("x")
        right.add("y")
        left.write("alpha", timestamp=5)
        right.write("beta", timestamp=5)
        right.merge(left)
        left.merge(right)
        self.assertEqual(left.clock.to_dict(), right.clock.to_dict(),
                         "the two clocks did not converge")
        self.assertEqual(left.read(), "beta")
        self.assertEqual(right.read(), "beta")
        self.assertEqual(sorted(left.elements()), ["x", "y"])
        self.assertEqual(sorted(right.elements()), ["x", "y"])
        sync(left, right)
        self.assertEqual(left.snapshot(), right.snapshot())


class ObservedRemoveTests(unittest.TestCase):

    def test_add_and_remove_tags_are_tracked(self):
        tagged = ORSet()
        self.assertTrue(tagged.add("x", ("a", 1)))
        self.assertTrue(tagged.contains("x"))
        self.assertEqual(tagged.elements(), ("x",))
        self.assertEqual(tagged.live_tags("x"), (("a", 1),))
        self.assertFalse(tagged.add("x", ("a", 1)))
        self.assertEqual(tagged.live_tags("x"), (("a", 1),))
        self.assertEqual(tagged.remove("x"), (("a", 1),))
        self.assertFalse(tagged.contains("x"), "remove left the element live")
        self.assertEqual(tagged.elements(), ())
        self.assertEqual(tagged.live_tags("x"), ())
        self.assertEqual(tagged.tombstone_tags("x"), (("a", 1),))
        self.assertFalse(tagged.add("x", ("a", 1)),
                         "a tombstoned tag came back")
        self.assertTrue(tagged.add("x", ("a", 2)))
        self.assertEqual(tagged.elements(), ("x",))

    def test_removed_element_stays_removed_after_a_merge(self):
        left = Replica("a")
        right = Replica("b")
        left.add("x")
        right.merge(left)
        right.remove("x")
        self.assertFalse(right.contains("x"), "remove left the element live")
        self.assertEqual(right.elements(), ())
        right.merge(left)
        self.assertFalse(right.contains("x"),
                         "a stale add brought the element back")
        left.merge(right)
        self.assertFalse(left.contains("x"))
        self.assertEqual(left.elements(), ())
        self.assertEqual(sorted(right.set.live_tags("x")), [])

    def test_pruning_drops_tombstones_up_to_the_stable_frontier(self):
        tagged = ORSet()
        for counter in (1, 3, 5):
            tagged.add("x", ("a", counter))
        tagged.remove("x")
        self.assertEqual(tagged.elements(), ())
        self.assertEqual(tagged.prune_tombstones({"a": 3}), 2)
        self.assertEqual(tagged.tombstone_tags("x"), (("a", 5),))
        self.assertEqual(tagged.prune_tombstones({"a": 5}), 1)
        self.assertEqual(tagged.tombstone_tags("x"), ())
        self.assertEqual(tagged.elements(), ())


class OperationIntakeTests(unittest.TestCase):

    def test_distinct_operations_from_different_nodes_survive(self):
        first = Replica("a")
        second = Replica("b")
        third = Replica("c")
        add_x = first.add("x")
        add_y = second.add("y")
        accepted = third.apply_ops([add_x, add_y, dict(add_x)])
        self.assertEqual(accepted, 2)
        self.assertEqual(sorted(third.elements()), ["x", "y"])
        self.assertEqual(third.duplicates, 1)
        self.assertEqual(third.clock.to_dict(), {"a": 1, "b": 1})

    def test_malformed_operations_are_rejected(self):
        replica = Replica("a")
        self.assertEqual(replica.apply_ops([None, {"kind": "nope"},
                                            {"kind": "add"}]), 0)
        self.assertFalse(replica.apply_op({"node": "a", "counter": 0,
                                           "kind": "add"}))
        self.assertEqual(replica.elements(), ())
        self.assertEqual(replica.clock.to_dict(), {})
        self.assertEqual(replica.apply_ops([{"node": "a", "counter": 1,
                                             "kind": "add",
                                             "element": "x"}]), 1)
        self.assertEqual(replica.elements(), ("x",))
        self.assertEqual(replica.clock.to_dict(), {"a": 1})


if __name__ == "__main__":
    unittest.main()
