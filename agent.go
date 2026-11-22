package main

import (
	"context"
	"encoding/json"
	"strings"
)

type ExtractionResult struct {
	Updates []MemoryUpdate
	Request []Message
}

type AnswerResult struct {
	Reply   string
	Request []Message
}

func ExtractMemory(ctx context.Context, llm LLMClient, user string, facts []*Fact, keys []*KeyPolicy) (ExtractionResult, Usage, error) {
	messages := buildExtractionMessages(user, facts, keys)
	response, err := llm.Chat(ctx, messages)
	if err != nil {
		return ExtractionResult{}, Usage{}, err
	}
	result := parseExtraction(response.Message.Content)
	result.Request = append([]Message{}, messages...)
	return result, response.Usage, nil
}

func AskAnswer(ctx context.Context, llm LLMClient, user string, short []Message, facts []*Fact) (AnswerResult, Usage, error) {
	messages := buildAnswerMessages(user, short, facts)
	response, err := llm.Chat(ctx, messages)
	if err != nil {
		return AnswerResult{}, Usage{}, err
	}
	return AnswerResult{Reply: strings.TrimSpace(response.Message.Content), Request: append([]Message{}, messages...)}, response.Usage, nil
}

func buildExtractionMessages(user string, facts []*Fact, keys []*KeyPolicy) []Message {
	working := formatFacts(activeFacts(facts, "working"))
	long := formatFacts(activeFacts(facts, "long"))
	var b strings.Builder
	b.WriteString("Ты извлекаешь факты памяти из сообщения пользователя.\n")
	b.WriteString("Верни только JSON без markdown:\n")
	b.WriteString(`{"updates":[{"key":"deadline","value":"20 марта","mode":"replace","layer":"working","reason":"пользователь сдвинул срок"}]}`)
	b.WriteString("\n\nПравила:\n")
	b.WriteString("- replace: срок, статус, версия, выбранный подход, стек, ответственный;\n")
	b.WriteString("- append: принятые решения, история решений, завершённые этапы;\n")
	b.WriteString("- delete: требование отменено или больше не действует;\n")
	b.WriteString("- working: факты текущей задачи; long: устойчивые предпочтения и решения, которые должны пережить завершение задачи;\n")
	b.WriteString("- если новых фактов нет, верни {\"updates\":[]}.\n")
	b.WriteString("- не включай старые значения в updates.\n\n")
	b.WriteString("Текущая working-память:\n" + emptyAsNone(working) + "\n")
	b.WriteString("Текущая long-term память:\n" + emptyAsNone(long) + "\n")
	b.WriteString("Известные ключи и их режимы:\n")
	if len(keys) == 0 {
		b.WriteString("(нет)\n")
	} else {
		for _, policy := range keys {
			b.WriteString("- " + policy.Key + ": " + policy.Mode + "/" + policy.Layer + "\n")
		}
	}
	b.WriteString("Для нового ключа выбери mode: replace, если это одно актуальное значение; append, если это элемент накапливаемого списка. Для существующего ключа не меняй mode.\n")
	b.WriteString("Сообщение пользователя:\n" + user)
	return []Message{{Role: "system", Content: b.String()}}
}

func buildAnswerMessages(user string, short []Message, facts []*Fact) []Message {
	working := formatFacts(selectFacts(facts, "working", user))
	long := formatFacts(selectFacts(facts, "long", user))
	var b strings.Builder
	b.WriteString("Ты — ассистент с памятью. Отвечай по актуальным фактам.\n")
	b.WriteString("В рабочей и долговременной памяти нет устаревших значений: используй только переданные факты.\n\n")
	b.WriteString("## Long-term memory\n" + emptyAsNone(long) + "\n")
	b.WriteString("## Working memory\n" + emptyAsNone(working) + "\n")
	b.WriteString("## Short-term memory\n")
	if len(short) == 0 {
		b.WriteString("(нет)\n")
	}
	messages := []Message{{Role: "system", Content: b.String()}}
	messages = append(messages, short...)
	return messages
}

func emptyAsNone(text string) string {
	if strings.TrimSpace(text) == "" {
		return "(нет)\n"
	}
	return text
}

func parseExtraction(content string) ExtractionResult {
	content = strings.TrimSpace(content)
	start := strings.Index(content, "{")
	end := strings.LastIndex(content, "}")
	if start < 0 || end < start {
		return ExtractionResult{Updates: []MemoryUpdate{}}
	}
	var parsed struct {
		Updates []MemoryUpdate `json:"updates"`
	}
	if err := json.Unmarshal([]byte(content[start:end+1]), &parsed); err != nil {
		return ExtractionResult{Updates: []MemoryUpdate{}}
	}
	if parsed.Updates == nil {
		parsed.Updates = []MemoryUpdate{}
	}
	return ExtractionResult{Updates: parsed.Updates}
}
