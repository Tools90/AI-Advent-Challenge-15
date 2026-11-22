package main

import (
	"bufio"
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"
)

//go:embed web/index.html
var webFS embed.FS

type config struct {
	keyFile string
	apiKey  string
	baseURL string
	model   string
	addr    string
}

type server struct {
	cfg    config
	memory *MemoryStore
	chat   *ChatStore
	llm    LLMClient
	mu     sync.Mutex
}

func main() {
	if err := loadDotEnv(".env"); err != nil {
		log.Printf("config: %v", err)
	}
	cfg := loadConfig()
	if cfg.apiKey == "" {
		log.Printf("deepseek: ключ не найден в %s", cfg.keyFile)
	}
	memory, err := NewMemoryStore("data")
	if err != nil {
		log.Fatalf("memory: %v", err)
	}
	chat, err := NewChatStore("data")
	if err != nil {
		log.Fatalf("chat: %v", err)
	}
	s := &server{
		cfg:    cfg,
		memory: memory,
		chat:   chat,
		llm:    NewDeepSeekClient(cfg.baseURL, cfg.apiKey, cfg.model),
	}
	srv := &http.Server{Addr: cfg.addr, Handler: s.handler(), ReadHeaderTimeout: 10 * time.Second}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		log.Printf("сервер запущен: http://localhost%s (модель: %s)", cfg.addr, cfg.model)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("server: %v", err)
		}
	}()
	<-ctx.Done()
	log.Println("остановка...")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("shutdown: %v", err)
	}
}

func (s *server) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /", s.index)
	mux.HandleFunc("GET /healthz", s.health)
	mux.HandleFunc("GET /api/state", s.state)
	mux.HandleFunc("POST /api/chat", s.chatHandler)
	mux.HandleFunc("POST /api/task/complete", s.completeTask)
	mux.HandleFunc("POST /api/clear-all", s.clearAll)
	return logRequests(mux)
}

func (s *server) index(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	data, err := webFS.ReadFile("web/index.html")
	if err != nil {
		http.Error(w, "index.html не найден", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(data)
}

func (s *server) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "model": s.cfg.model, "configured": s.cfg.apiKey != ""})
}

func (s *server) state(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.statePayload())
}

func (s *server) statePayload() map[string]any {
	memory := s.memory.State()
	chat := s.chat.State()
	return map[string]any{
		"model":      s.cfg.model,
		"configured": s.cfg.apiKey != "",
		"facts":      memory.Facts,
		"logs":       memory.Logs,
		"keys":       memory.Keys,
		"chat":       chat.Messages,
		"turns":      chat.Turns,
	}
}

func (s *server) chatHandler(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Message string `json:"message"`
	}
	if err := decodeJSON(w, r, &body); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	message := strings.TrimSpace(body.Message)
	if message == "" {
		writeError(w, http.StatusBadRequest, "сообщение пустое")
		return
	}
	if len([]rune(message)) > 4000 {
		writeError(w, http.StatusBadRequest, "сообщение слишком длинное")
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	chat := s.chat.State()
	userIndex := len(chat.Messages)
	if err := s.chat.AddMessage(Message{Role: "user", Content: message}); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	short := s.chat.Recent(5)
	facts := s.memory.State().Facts
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()

	extraction, memoryUsage, extractionErr := ExtractMemory(ctx, s.llm, message, facts, s.memory.State().Keys)
	var appliedLogs []MemoryLog
	if extractionErr == nil && len(extraction.Updates) > 0 {
		appliedLogs, _ = s.memory.ApplyUpdates(extraction.Updates)
	}
	facts = s.memory.State().Facts
	answer, answerUsage, err := AskAnswer(ctx, s.llm, message, short, facts)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	if answer.Reply == "" {
		answer.Reply = "Не удалось сформировать ответ."
	}
	assistantIndex := len(s.chat.State().Messages)
	if err := s.chat.AddMessage(Message{Role: "assistant", Content: answer.Reply}); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	_ = s.chat.AddTurn(TurnMeta{
		UserIndex:      userIndex,
		AssistantIndex: assistantIndex,
		MemoryRequest:  extraction.Request,
		AnswerRequest:  answer.Request,
	})
	writeJSON(w, http.StatusOK, map[string]any{
		"reply":           answer.Reply,
		"updates":         extraction.Updates,
		"logs":            appliedLogs,
		"memoryUsage":     memoryUsage,
		"answerUsage":     answerUsage,
		"extractionError": errorString(extractionErr),
		"state":           s.statePayload(),
	})
}

func (s *server) completeTask(w http.ResponseWriter, r *http.Request) {
	if err := s.memory.CompleteTask(); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"state": s.statePayload()})
}

func (s *server) clearAll(w http.ResponseWriter, r *http.Request) {
	if err := s.memory.ClearAll(); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if err := s.chat.Clear(); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"state": s.statePayload()})
}

func errorString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func loadConfig() config {
	keyFile := envOr("DEEPSEEK_API_KEY_FILE", "api_key.txt")
	port := envOr("PORT", "8095")
	return config{
		keyFile: keyFile,
		apiKey:  readSecret(keyFile, "sk-your-key-here"),
		baseURL: strings.TrimRight(envOr("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), "/"),
		model:   envOr("DEEPSEEK_MODEL", "deepseek-flash"),
		addr:    ":" + port,
	}
}

func readSecret(path, placeholder string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	value := unquote(strings.TrimSpace(strings.TrimPrefix(string(data), "\ufeff")))
	if value == placeholder || strings.Contains(value, "your-") {
		return ""
	}
	return value
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func decodeJSON(w http.ResponseWriter, r *http.Request, dst any) error {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	if err := json.NewDecoder(r.Body).Decode(dst); err != nil {
		return fmt.Errorf("неверный JSON: %w", err)
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond))
	})
}

func loadDotEnv(path string) error {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("файл %s не найден, используем переменные окружения", path)
		}
		return err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(strings.TrimPrefix(scanner.Text(), "\ufeff"))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(strings.TrimPrefix(key, "export "))
		value = unquote(strings.TrimSpace(value))
		if key == "" {
			continue
		}
		if _, exists := os.LookupEnv(key); !exists {
			_ = os.Setenv(key, value)
		}
	}
	return scanner.Err()
}

func unquote(v string) string {
	if len(v) >= 2 {
		if (v[0] == '"' && v[len(v)-1] == '"') || (v[0] == '\'' && v[len(v)-1] == '\'') {
			return v[1 : len(v)-1]
		}
	}
	return v
}
