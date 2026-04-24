package chameleon

type radixNode struct {
	children map[int8]*radixNode
	traceIDs []int
}

func newRadixNode() *radixNode {
	return &radixNode{
		children: make(map[int8]*radixNode),
		traceIDs: make([]int, 0, 4),
	}
}

type radixTrie struct {
	root *radixNode
}

func newRadixTrie() *radixTrie {
	return &radixTrie{root: newRadixNode()}
}

func (t *radixTrie) Insert(seq []int8, traceID int) {
	if t == nil || t.root == nil {
		return
	}
	n := t.root
	n.traceIDs = append(n.traceIDs, traceID)
	for _, v := range seq {
		child := n.children[v]
		if child == nil {
			child = newRadixNode()
			n.children[v] = child
		}
		n = child
		n.traceIDs = append(n.traceIDs, traceID)
	}
}

func (t *radixTrie) MatchPrefix(prefix []int8) []int {
	if t == nil || t.root == nil {
		return nil
	}
	n := t.root
	for _, v := range prefix {
		child := n.children[v]
		if child == nil {
			return nil
		}
		n = child
	}
	out := make([]int, len(n.traceIDs))
	copy(out, n.traceIDs)
	return out
}
