package chameleon // import "github.com/websitefingerprinting/wfdef.git/transports/chameleon"

import (
	"encoding/json"
	"io"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"

	pt "git.torproject.org/pluggable-transports/goptlib.git"
	"github.com/websitefingerprinting/wfdef.git/common/log"
	"github.com/websitefingerprinting/wfdef.git/common/utils"
	"github.com/websitefingerprinting/wfdef.git/transports/base"
	"github.com/websitefingerprinting/wfdef.git/transports/defconn"
)

const (
	transportName      = "chameleon"
	chameleonDataFile  = "ds-19.json"
	maxPrefixLen       = 96
	defaultMutationLen = 8
)

type tracePoint struct {
	Timestamp float64
	Direction int8
}

type traceDataset struct {
	SelectedTraces [][][]float64 `json:"selected_traces"`
	IdxSet         []int         `json:"idx_set"`
}

type traceTemplate struct {
	idx    int
	points []tracePoint
}

type traceEngine struct {
	mu        sync.RWMutex
	templates []traceTemplate
	idxSet    []int
	trie      *radixTrie
	rng       *rand.Rand
}

type selectedTemplate struct {
	id     int
	points []tracePoint
}

func loadTraceEngine() (*traceEngine, error) {
	exeDir := stateDirForExecutable()
	candidates := []string{
		filepath.Join(exeDir, chameleonDataFile),
		filepath.Join("transports", "chameleon", chameleonDataFile),
		filepath.Join(exeDir, "transports", "chameleon", chameleonDataFile),
		filepath.Join(exeDir, "..", "transports", "chameleon", chameleonDataFile),
		filepath.Join(exeDir, "..", "..", "transports", "chameleon", chameleonDataFile),
	}

	var raw []byte
	var readErr error
	for _, p := range candidates {
		raw, readErr = os.ReadFile(p)
		if readErr == nil {
			break
		}
	}
	if readErr != nil {
		return nil, readErr
	}
	ds := traceDataset{}
	if err := json.Unmarshal(raw, &ds); err != nil {
		return nil, err
	}
	engine := &traceEngine{
		templates: make([]traceTemplate, 0, len(ds.SelectedTraces)),
		idxSet:    ds.IdxSet,
		trie:      newRadixTrie(),
		rng:       rand.New(rand.NewSource(time.Now().UnixNano())),
	}
	for i, tr := range ds.SelectedTraces {
		points := make([]tracePoint, 0, len(tr))
		dirs := make([]int8, 0, len(tr))
		for _, p := range tr {
			if len(p) < 2 {
				continue
			}
			dir := int8(1)
			if p[1] < 0 {
				dir = -1
			}
			points = append(points, tracePoint{Timestamp: p[0], Direction: dir})
			dirs = append(dirs, dir)
		}
		if len(points) == 0 {
			continue
		}
		engine.templates = append(engine.templates, traceTemplate{idx: i, points: points})
		engine.trie.Insert(dirs, i)
	}
	if len(engine.templates) == 0 {
		return nil, io.EOF
	}
	return engine, nil
}

func stateDirForExecutable() string {
	exePath, err := os.Executable()
	if err != nil {
		return "."
	}
	return filepath.Dir(exePath)
}

func (e *traceEngine) pickTemplate(prefix []int8, outbound int8) selectedTemplate {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if len(e.templates) == 0 {
		return selectedTemplate{id: -1}
	}
	candidates := e.trie.MatchPrefix(prefix)
	selected := -1
	if len(candidates) > 0 {
		selected = candidates[e.rng.Intn(len(candidates))]
	}
	if selected < 0 || selected >= len(e.templates) {
		selected = e.rng.Intn(len(e.templates))
	}
	ref := e.templates[selected].points
	return selectedTemplate{
		id:     selected,
		points: mutateTrace(ref, prefix, outbound, defaultMutationLen),
	}
}

func (e *traceEngine) templateByID(templateID int, prefix []int8, outbound int8) []tracePoint {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if templateID < 0 || templateID >= len(e.templates) {
		return nil
	}
	ref := e.templates[templateID].points
	return mutateTrace(ref, prefix, outbound, defaultMutationLen)
}

func mutateTrace(ref []tracePoint, prefix []int8, outbound int8, mutationLen int) []tracePoint {
	mut := make([]tracePoint, len(ref))
	copy(mut, ref)
	start := len(prefix)
	if start >= len(mut) {
		return mut
	}
	end := start + mutationLen
	if end > len(mut) {
		end = len(mut)
	}
	hasOutbound := false
	for i := start; i < end; i++ {
		if mut[i].Direction == outbound {
			hasOutbound = true
			break
		}
	}
	if !hasOutbound {
		mut[start].Direction = outbound
	}
	return mut
}

type Transport struct {
	defconn.Transport
	once   sync.Once
	engine *traceEngine
	err    error
}

func (t *Transport) Name() string {
	return transportName
}

func (t *Transport) ensureEngine() error {
	t.once.Do(func() {
		t.engine, t.err = loadTraceEngine()
	})
	return t.err
}

func (t *Transport) ClientFactory(stateDir string) (base.ClientFactory, error) {
	if err := t.ensureEngine(); err != nil {
		return nil, err
	}
	parentFactory, err := t.Transport.ClientFactory(stateDir)
	if err != nil {
		return nil, err
	}
	return &chameleonClientFactory{
		DefConnClientFactory: parentFactory.(*defconn.DefConnClientFactory),
		engine:               t.engine,
	}, nil
}

func (t *Transport) ServerFactory(stateDir string, args *pt.Args) (base.ServerFactory, error) {
	if err := t.ensureEngine(); err != nil {
		return nil, err
	}
	sf, err := t.Transport.ServerFactory(stateDir, args)
	if err != nil {
		return nil, err
	}
	return &chameleonServerFactory{
		DefConnServerFactory: sf.(*defconn.DefConnServerFactory),
		engine:               t.engine,
	}, nil
}

type chameleonClientFactory struct {
	*defconn.DefConnClientFactory
	engine *traceEngine
}

func (cf *chameleonClientFactory) Transport() base.Transport {
	return cf.DefConnClientFactory.Transport()
}

func (cf *chameleonClientFactory) Dial(network, addr string, dialFn base.DialFunc, args interface{}) (net.Conn, error) {
	defConn, err := cf.DefConnClientFactory.Dial(network, addr, dialFn, args)
	if err != nil {
		return nil, err
	}
	return &chameleonConn{
		DefConn:     defConn.(*defconn.DefConn),
		engine:      cf.engine,
		outboundDir: 1,
		coordEpoch:  uint32(time.Now().UnixNano()),
	}, nil
}

type chameleonServerFactory struct {
	*defconn.DefConnServerFactory
	engine *traceEngine
}

func (sf *chameleonServerFactory) WrapConn(conn net.Conn) (net.Conn, error) {
	defConn, err := sf.DefConnServerFactory.WrapConn(conn)
	if err != nil {
		return nil, err
	}
	return &chameleonConn{
		DefConn:     defConn.(*defconn.DefConn),
		engine:      sf.engine,
		outboundDir: -1,
		coordEpoch:  uint32(time.Now().UnixNano()),
	}, nil
}

type chameleonConn struct {
	*defconn.DefConn
	engine      *traceEngine
	outboundDir int8

	coordMu       sync.Mutex
	coordEpoch    uint32
	localSeq      uint32
	lastRemoteSeq uint32
	pendingSelect selectedTemplate
	hasPendingSel bool
}

func (conn *chameleonConn) nextTemplate(prefix []int8) selectedTemplate {
	if len(prefix) > maxPrefixLen {
		prefix = prefix[len(prefix)-maxPrefixLen:]
	}
	return conn.engine.pickTemplate(prefix, conn.outboundDir)
}

func (conn *chameleonConn) signalTraceSelection(templateID int, startPos uint16) {
	conn.coordMu.Lock()
	conn.localSeq++
	seq := conn.localSeq
	epoch := conn.coordEpoch
	conn.coordMu.Unlock()

	conn.SendTraceSelect(defconn.TraceSignalMsg{
		Epoch:    epoch,
		Seq:      seq,
		TraceID:  uint32(templateID),
		StartPos: startPos,
	})
}

func (conn *chameleonConn) signalTraceAck(msg defconn.TraceSignalMsg) {
	conn.SendTraceAck(defconn.TraceSignalMsg{
		Epoch:    msg.Epoch,
		Seq:      msg.Seq,
		TraceID:  msg.TraceID,
		StartPos: msg.StartPos,
	})
}

func (conn *chameleonConn) drainTraceSignals(prefix []int8) {
	for {
		select {
		case msg := <-conn.TraceSignalChan:
			if msg.IsAck {
				continue
			}

			conn.coordMu.Lock()
			if msg.Epoch != conn.coordEpoch || msg.Seq <= conn.lastRemoteSeq {
				conn.coordMu.Unlock()
				conn.signalTraceAck(msg)
				continue
			}
			conn.lastRemoteSeq = msg.Seq
			conn.coordMu.Unlock()

			selected := conn.engine.templateByID(int(msg.TraceID), prefix, conn.outboundDir)
			if len(selected) == 0 {
				conn.signalTraceAck(msg)
				continue
			}
			start := int(msg.StartPos)
			if start >= len(selected) {
				start = 0
			}
			conn.coordMu.Lock()
			conn.pendingSelect = selectedTemplate{
				id:     int(msg.TraceID),
				points: selected[start:],
			}
			conn.hasPendingSel = true
			conn.coordMu.Unlock()
			conn.signalTraceAck(msg)
		default:
			return
		}
	}
}

func (conn *chameleonConn) consumePendingTemplate() (selectedTemplate, bool) {
	conn.coordMu.Lock()
	defer conn.coordMu.Unlock()
	if !conn.hasPendingSel {
		return selectedTemplate{}, false
	}
	sel := conn.pendingSelect
	conn.pendingSelect = selectedTemplate{}
	conn.hasPendingSel = false
	return sel, true
}

func (conn *chameleonConn) ReadFrom(r io.Reader) (written int64, err error) {
	defer close(conn.CloseChan)
	go conn.Send()

	var receiveBuf utils.SafeBuffer
	doneRead := make(chan error, 1)

	go func() {
		for {
			buf := make([]byte, 65535)
			rdLen, rdErr := r.Read(buf)
			if rdErr != nil {
				doneRead <- rdErr
				return
			}
			if rdLen <= 0 {
				doneRead <- io.EOF
				return
			}
			if _, wErr := receiveBuf.Write(buf[:rdLen]); wErr != nil {
				doneRead <- wErr
				return
			}
			if !conn.IsServer {
				state := conn.ConnState.LoadCurState()
				if (state == defconn.StateStop && rdLen > defconn.MaxPacketPayloadLength) || state == defconn.StateReady {
					conn.ConnState.SetState(defconn.StateStart)
					conn.SendChan <- defconn.PacketInfo{PktType: defconn.PacketTypeSignalStart, Data: []byte{}, PadLen: defconn.MaxPacketPaddingLength}
				} else if state == defconn.StateStop {
					conn.ConnState.SetState(defconn.StateReady)
				}
			}
		}
	}()

	initial := conn.nextTemplate(nil)
	if len(initial.points) == 0 {
		return 0, io.EOF
	}
	template := initial.points
	conn.signalTraceSelection(initial.id, 0)
	pos := 0
	lastSend := time.Now()
	prefix := make([]int8, 0, maxPrefixLen)

	for {
		select {
		case conErr := <-conn.ErrChan:
			return written, conErr
		case rdErr := <-doneRead:
			return written, rdErr
		default:
			conn.drainTraceSignals(prefix)
			if remoteSel, ok := conn.consumePendingTemplate(); ok {
				template = remoteSel.points
				pos = 0
			}
			if pos >= len(template) {
				nextSel := conn.nextTemplate(prefix)
				template = nextSel.points
				pos = 0
				if len(template) == 0 {
					return written, io.EOF
				}
				conn.signalTraceSelection(nextSel.id, 0)
			}

			delay := time.Duration(0)
			if pos > 0 {
				delta := template[pos].Timestamp - template[pos-1].Timestamp
				if delta > 0 {
					delay = time.Duration(delta * float64(time.Second))
				}
			}
			if delay > 0 {
				utils.SleepRho(lastSend, delay)
			}
			lastSend = time.Now()

			dir := template[pos].Direction
			pos++
			if len(prefix) >= maxPrefixLen {
				copy(prefix, prefix[1:])
				prefix[len(prefix)-1] = dir
			} else {
				prefix = append(prefix, dir)
			}
			if dir != conn.outboundDir {
				continue
			}

			if receiveBuf.GetLen() > 0 {
				var payload [defconn.MaxPacketPayloadLength]byte
				rdLen, rdErr := receiveBuf.Read(payload[:])
				if rdErr != nil {
					return written, rdErr
				}
				written += int64(rdLen)
				conn.SendChan <- defconn.PacketInfo{
					PktType: defconn.PacketTypePayload,
					Data:    payload[:rdLen],
					PadLen:  uint16(defconn.MaxPacketPaddingLength - rdLen),
				}
				conn.NRealSegSentIncrement()
				continue
			}

			state := conn.ConnState.LoadCurState()
			if state == defconn.StateStop || state == defconn.StateReady {
				continue
			}
			conn.SendChan <- defconn.PacketInfo{
				PktType: defconn.PacketTypeDummy,
				Data:    []byte{},
				PadLen:  defconn.MaxPacketPaddingLength,
			}
		}
	}
}

func (conn *chameleonConn) Read(b []byte) (n int, err error) {
	return conn.DefConn.MyRead(b, conn.readPackets)
}

func (conn *chameleonConn) readPackets() error {
	err := conn.DefConn.ReadPackets()
	if err != nil {
		return err
	}
	if !conn.IsServer {
		return nil
	}
	state := conn.ConnState.LoadCurState()
	if state == defconn.StateStart {
		log.Debugf("[Chameleon][%s] Server state start", conn.RemoteAddr())
	}
	return nil
}

var _ base.ClientFactory = (*chameleonClientFactory)(nil)
var _ base.ServerFactory = (*chameleonServerFactory)(nil)
var _ base.Transport = (*Transport)(nil)
var _ net.Conn = (*chameleonConn)(nil)
