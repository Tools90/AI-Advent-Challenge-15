package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

type TurnMeta struct {
	UserIndex      int       `json:"userIndex"`
	AssistantIndex int       `json:"assistantIndex"`
	MemoryRequest  []Message `json:"memoryRequest"`
	AnswerRequest  []Message `json:"answerRequest"`
}

type ChatState struct {
	Messages []Message  `json:"messages"`
	Turns    []TurnMeta `json:"turns"`
}

type ChatStore struct {
	mu    sync.RWMutex
	path  string
	state ChatState
}

func NewChatStore(dir string) (*ChatStore, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("создать каталог %s: %w", dir, err)
	}
	store := &ChatStore{path: filepath.Join(dir, "chat.json")}
	data, err := os.ReadFile(store.path)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("прочитать %s: %w", store.path, err)
		}
		store.state = ChatState{Messages: []Message{}, Turns: []TurnMeta{}}
		if err := store.saveLocked(); err != nil {
			return nil, err
		}
		return store, nil
	}
	data = trimBOM(data)
	if err := json.Unmarshal(data, &store.state); err != nil {
		return nil, fmt.Errorf("разобрать %s: %w", store.path, err)
	}
	if store.state.Messages == nil {
		store.state.Messages = []Message{}
	}
	if store.state.Turns == nil {
		store.state.Turns = []TurnMeta{}
	}
	return store, nil
}

func (s *ChatStore) State() ChatState {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return ChatState{Messages: append([]Message{}, s.state.Messages...), Turns: cloneTurns(s.state.Turns)}
}

func (s *ChatStore) AddMessage(message Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Messages = append(s.state.Messages, message)
	return s.saveLocked()
}

func (s *ChatStore) AddTurn(turn TurnMeta) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Turns = append(s.state.Turns, turn)
	return s.saveLocked()
}

func (s *ChatStore) Recent(limit int) []Message {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if limit <= 0 || len(s.state.Messages) <= limit {
		return append([]Message{}, s.state.Messages...)
	}
	return append([]Message{}, s.state.Messages[len(s.state.Messages)-limit:]...)
}

func (s *ChatStore) Clear() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state = ChatState{Messages: []Message{}, Turns: []TurnMeta{}}
	if err := os.Remove(s.path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

func (s *ChatStore) saveLocked() error {
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

func cloneTurns(turns []TurnMeta) []TurnMeta {
	out := make([]TurnMeta, len(turns))
	for i, turn := range turns {
		turn.MemoryRequest = append([]Message{}, turn.MemoryRequest...)
		turn.AnswerRequest = append([]Message{}, turn.AnswerRequest...)
		out[i] = turn
	}
	return out
}
