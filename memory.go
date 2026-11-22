package main

import (
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

type Fact struct {
	ID        string `json:"id"`
	Key       string `json:"key"`
	Value     string `json:"value"`
	Layer     string `json:"layer"`
	Mode      string `json:"mode"`
	Status    string `json:"status"`
	Pinned    bool   `json:"pinned"`
	CreatedAt string `json:"createdAt"`
	UpdatedAt string `json:"updatedAt"`
	Source    string `json:"source"`
}

type MemoryLog struct {
	ID       string `json:"id"`
	Key      string `json:"key"`
	Layer    string `json:"layer"`
	Action   string `json:"action"`
	OldValue string `json:"oldValue"`
	NewValue string `json:"newValue"`
	Reason   string `json:"reason"`
	At       string `json:"at"`
}

type MemoryState struct {
	Facts []*Fact      `json:"facts"`
	Logs  []MemoryLog  `json:"logs"`
	Keys  []*KeyPolicy `json:"keys"`
}

type KeyPolicy struct {
	Key       string `json:"key"`
	Mode      string `json:"mode"`
	Layer     string `json:"layer"`
	CreatedAt string `json:"createdAt"`
	UpdatedAt string `json:"updatedAt"`
}

type MemoryUpdate struct {
	Key    string `json:"key"`
	Value  string `json:"value"`
	Mode   string `json:"mode"`
	Layer  string `json:"layer"`
	Reason string `json:"reason"`
	Pinned bool   `json:"pinned"`
}

type MemoryStore struct {
	mu    sync.RWMutex
	path  string
	state MemoryState
}

func NewMemoryStore(dir string) (*MemoryStore, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("создать каталог %s: %w", dir, err)
	}
	store := &MemoryStore{path: filepath.Join(dir, "memory.json")}
	data, err := os.ReadFile(store.path)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("прочитать %s: %w", store.path, err)
		}
		store.state = MemoryState{Facts: []*Fact{}, Logs: []MemoryLog{}, Keys: []*KeyPolicy{}}
		if err := store.saveLocked(); err != nil {
			return nil, err
		}
		return store, nil
	}
	data = trimBOM(data)
	if err := json.Unmarshal(data, &store.state); err != nil {
		return nil, fmt.Errorf("разобрать %s: %w", store.path, err)
	}
	normalizeMemoryState(&store.state)
	if err := store.saveLocked(); err != nil {
		return nil, err
	}
	return store, nil
}

func (s *MemoryStore) State() MemoryState {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneMemoryState(s.state)
}

func (s *MemoryStore) ApplyUpdates(updates []MemoryUpdate) ([]MemoryLog, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now().Format(time.RFC3339Nano)
	applied := []MemoryLog{}
	for _, update := range updates {
		update.Key = strings.TrimSpace(strings.ToLower(update.Key))
		update.Value = strings.TrimSpace(update.Value)
		update.Mode = strings.ToLower(strings.TrimSpace(update.Mode))
		update.Layer = strings.ToLower(strings.TrimSpace(update.Layer))
		if update.Mode == "" {
			update.Mode = "replace"
		}
		if update.Layer == "" {
			update.Layer = "working"
		}
		if update.Key == "" {
			return applied, fmt.Errorf("у факта нет key")
		}
		if update.Layer != "working" && update.Layer != "long" {
			return applied, fmt.Errorf("неизвестный слой %s", update.Layer)
		}
		if update.Mode != "replace" && update.Mode != "append" && update.Mode != "delete" {
			return applied, fmt.Errorf("неизвестный режим %s", update.Mode)
		}
		if update.Mode != "delete" && update.Value == "" {
			return applied, fmt.Errorf("у факта нет значения")
		}
		policy := findKeyPolicy(s.state.Keys, update.Key)
		if policy != nil {
			update.Layer = policy.Layer
			if update.Mode != "delete" {
				update.Mode = policy.Mode
			}
		} else if update.Mode != "delete" {
			policy = &KeyPolicy{Key: update.Key, Mode: update.Mode, Layer: update.Layer, CreatedAt: now, UpdatedAt: now}
			s.state.Keys = append(s.state.Keys, policy)
		}
		switch update.Mode {
		case "replace":
			oldValues := s.archiveActiveLocked(update.Key, update.Layer, now)
			fact := &Fact{
				ID: newID(), Key: update.Key, Value: update.Value, Layer: update.Layer,
				Mode: "replace", Status: "active", Pinned: update.Pinned,
				CreatedAt: now, UpdatedAt: now, Source: "chat",
			}
			s.state.Facts = append(s.state.Facts, fact)
			applied = append(applied, MemoryLog{ID: newID(), Key: update.Key, Layer: update.Layer, Action: "replace", OldValue: strings.Join(oldValues, ", "), NewValue: update.Value, Reason: update.Reason, At: now})
		case "append":
			if hasActiveValue(s.state.Facts, update.Key, update.Value, update.Layer) {
				continue
			}
			fact := &Fact{
				ID: newID(), Key: update.Key, Value: update.Value, Layer: update.Layer,
				Mode: "append", Status: "active", Pinned: update.Pinned,
				CreatedAt: now, UpdatedAt: now, Source: "chat",
			}
			s.state.Facts = append(s.state.Facts, fact)
			applied = append(applied, MemoryLog{ID: newID(), Key: update.Key, Layer: update.Layer, Action: "append", NewValue: update.Value, Reason: update.Reason, At: now})
		case "delete":
			old := s.archiveActiveLocked(update.Key, update.Layer, now)
			applied = append(applied, MemoryLog{ID: newID(), Key: update.Key, Layer: update.Layer, Action: "delete", OldValue: strings.Join(old, ", "), Reason: update.Reason, At: now})
		}
	}
	s.state.Logs = append(s.state.Logs, applied...)
	if err := s.saveLocked(); err != nil {
		return applied, err
	}
	return applied, nil
}

func (s *MemoryStore) CompleteTask() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now().Format(time.RFC3339Nano)
	for _, fact := range s.state.Facts {
		if fact.Status == "active" && fact.Layer == "working" {
			fact.Status = "archived"
			fact.UpdatedAt = now
			s.state.Logs = append(s.state.Logs, MemoryLog{
				ID: newID(), Key: fact.Key, Layer: fact.Layer, Action: "clear_working",
				OldValue: fact.Value, NewValue: "", Reason: "задача завершена", At: now,
			})
		}
	}
	return s.saveLocked()
}

func (s *MemoryStore) ClearAll() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state = MemoryState{Facts: []*Fact{}, Logs: []MemoryLog{}, Keys: []*KeyPolicy{}}
	if err := os.Remove(s.path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

func (s *MemoryStore) archiveActiveLocked(key, layer, now string) []string {
	old := []string{}
	for _, fact := range s.state.Facts {
		if fact.Status == "active" && fact.Key == key && fact.Layer == layer {
			old = append(old, fact.Value)
			fact.Status = "archived"
			fact.UpdatedAt = now
		}
	}
	return old
}

func lastValue(facts []*Fact, key, layer, excludeID string) string {
	values := []string{}
	for _, fact := range facts {
		if fact.ID != excludeID && fact.Status == "active" && fact.Key == key && fact.Layer == layer {
			values = append(values, fact.Value)
		}
	}
	return strings.Join(values, ", ")
}

func hasActiveValue(facts []*Fact, key, value, layer string) bool {
	for _, fact := range facts {
		if fact.Status == "active" && fact.Key == key && fact.Value == value && fact.Layer == layer {
			return true
		}
	}
	return false
}

func findKeyPolicy(keys []*KeyPolicy, key string) *KeyPolicy {
	for _, policy := range keys {
		if policy.Key == key {
			return policy
		}
	}
	return nil
}

func (s *MemoryStore) saveLocked() error {
	data, err := json.MarshalIndent(s.state, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	_ = os.Remove(s.path)
	return os.Rename(tmp, s.path)
}

func normalizeMemoryState(state *MemoryState) {
	if state.Facts == nil {
		state.Facts = []*Fact{}
	}
	if state.Logs == nil {
		state.Logs = []MemoryLog{}
	}
	filtered := make([]MemoryLog, 0, len(state.Logs))
	for _, log := range state.Logs {
		if log.Action != "key_registered" {
			filtered = append(filtered, log)
		}
	}
	state.Logs = filtered
	if state.Keys == nil {
		state.Keys = []*KeyPolicy{}
	}
	for _, fact := range state.Facts {
		if fact.ID == "" {
			fact.ID = newID()
		}
		if fact.Mode == "" {
			fact.Mode = "replace"
		}
		if fact.Status == "" {
			fact.Status = "active"
		}
		if fact.Layer == "" {
			fact.Layer = "working"
		}
		if findKeyPolicy(state.Keys, fact.Key) == nil {
			state.Keys = append(state.Keys, &KeyPolicy{Key: fact.Key, Mode: fact.Mode, Layer: fact.Layer, CreatedAt: fact.CreatedAt, UpdatedAt: fact.UpdatedAt})
		}
	}
}

func cloneMemoryState(state MemoryState) MemoryState {
	copy := MemoryState{
		Facts: make([]*Fact, 0, len(state.Facts)),
		Logs:  append([]MemoryLog{}, state.Logs...),
		Keys:  make([]*KeyPolicy, 0, len(state.Keys)),
	}
	for _, fact := range state.Facts {
		item := *fact
		copy.Facts = append(copy.Facts, &item)
	}
	for _, policy := range state.Keys {
		item := *policy
		copy.Keys = append(copy.Keys, &item)
	}
	return copy
}

func activeFacts(facts []*Fact, layer string) []*Fact {
	out := []*Fact{}
	for _, fact := range facts {
		if fact.Status == "active" && fact.Layer == layer {
			out = append(out, fact)
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Pinned != out[j].Pinned {
			return out[i].Pinned
		}
		return out[i].UpdatedAt < out[j].UpdatedAt
	})
	return out
}

func selectFacts(facts []*Fact, layer, query string) []*Fact {
	items := activeFacts(facts, layer)
	if len(items) == 0 {
		return items
	}
	picked := []*Fact{}
	pinned := map[string]bool{}
	for _, fact := range items {
		if fact.Pinned {
			picked = append(picked, fact)
			pinned[fact.ID] = true
		}
	}
	words := strings.Fields(strings.ToLower(query))
	remaining := []*Fact{}
	for _, fact := range items {
		if pinned[fact.ID] {
			continue
		}
		score := 0
		text := strings.ToLower(fact.Key + " " + fact.Value)
		for _, word := range words {
			if len([]rune(word)) >= 3 && strings.Contains(text, word) {
				score++
			}
		}
		if score > 0 {
			remaining = append(remaining, fact)
		}
	}
	if len(remaining) == 0 {
		for _, fact := range items {
			if !pinned[fact.ID] {
				remaining = append(remaining, fact)
			}
		}
	}
	sort.SliceStable(remaining, func(i, j int) bool { return remaining[i].UpdatedAt > remaining[j].UpdatedAt })
	if len(remaining) > 3 {
		remaining = remaining[:3]
	}
	return append(picked, remaining...)
}

func formatFacts(facts []*Fact) string {
	if len(facts) == 0 {
		return ""
	}
	var b strings.Builder
	grouped := map[string][]*Fact{}
	order := []string{}
	for _, fact := range facts {
		if _, ok := grouped[fact.Key]; !ok {
			order = append(order, fact.Key)
		}
		grouped[fact.Key] = append(grouped[fact.Key], fact)
	}
	for _, key := range order {
		items := grouped[key]
		array := false
		for _, fact := range items {
			if fact.Mode == "append" {
				array = true
				break
			}
		}
		for i, fact := range items {
			label := key
			if array {
				label = fmt.Sprintf("%s[%d]", key, i)
			}
			b.WriteString("- " + label + ": " + fact.Value + "\n")
		}
	}
	return b.String()
}

func newID() string {
	buf := make([]byte, 8)
	if _, err := rand.Read(buf); err != nil {
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return fmt.Sprintf("%x", buf)
}

func trimBOM(data []byte) []byte {
	if len(data) >= 3 && data[0] == 0xEF && data[1] == 0xBB && data[2] == 0xBF {
		return data[3:]
	}
	return data
}
