package services

import (
	"fmt"
	"sync"
	"time"
)

// LogLevel represents the severity of a log entry.
type LogLevel int

const (
	LogLevelDebug LogLevel = iota
	LogLevelInfo
	LogLevelWarn
	LogLevelError
)

func (l LogLevel) String() string {
	switch l {
	case LogLevelDebug:
		return "DEBUG"
	case LogLevelInfo:
		return "INFO"
	case LogLevelWarn:
		return "WARN"
	case LogLevelError:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

// LogEntry represents a single structured log record.
type LogEntry struct {
	Timestamp time.Time         `json:"timestamp"`
	Level     LogLevel          `json:"level"`
	LevelStr  string            `json:"level_str"`
	Component string            `json:"component"`
	Message   string            `json:"message"`
	Fields    map[string]string `json:"fields,omitempty"`
}

// LogService provides structured, thread-safe logging for the desktop application.
type LogService struct {
	mu       sync.RWMutex
	entries  []LogEntry
	minLevel LogLevel
	maxSize  int
}

// NewLogService creates a new LogService with the given minimum log level.
func NewLogService(minLevel LogLevel) *LogService {
	return &LogService{
		entries:  make([]LogEntry, 0, 256),
		minLevel: minLevel,
		maxSize:  10000,
	}
}

func (ls *LogService) log(level LogLevel, component, message string, fields map[string]string) {
	if level < ls.minLevel {
		return
	}

	entry := LogEntry{
		Timestamp: time.Now(),
		Level:     level,
		LevelStr:  level.String(),
		Component: component,
		Message:   message,
		Fields:    fields,
	}

	ls.mu.Lock()
	defer ls.mu.Unlock()

	// Evict oldest entries if at capacity
	if len(ls.entries) >= ls.maxSize {
		ls.entries = ls.entries[1:]
	}
	ls.entries = append(ls.entries, entry)
}

// Debug logs a debug-level message.
func (ls *LogService) Debug(component, message string, fields map[string]string) {
	ls.log(LogLevelDebug, component, message, fields)
}

// Info logs an info-level message.
func (ls *LogService) Info(component, message string, fields map[string]string) {
	ls.log(LogLevelInfo, component, message, fields)
}

// Warn logs a warning-level message.
func (ls *LogService) Warn(component, message string, fields map[string]string) {
	ls.log(LogLevelWarn, component, message, fields)
}

// Error logs an error-level message.
func (ls *LogService) Error(component, message string, fields map[string]string) {
	ls.log(LogLevelError, component, message, fields)
}

// GetEntries returns log entries filtered by minimum level.
// Returns a copy to avoid data races.
func (ls *LogService) GetEntries(minLevel LogLevel) []LogEntry {
	ls.mu.RLock()
	defer ls.mu.RUnlock()

	result := make([]LogEntry, 0, len(ls.entries))
	for _, e := range ls.entries {
		if e.Level >= minLevel {
			result = append(result, e)
		}
	}
	return result
}

// GetRecentEntries returns the last n entries at or above the given level.
func (ls *LogService) GetRecentEntries(n int, minLevel LogLevel) []LogEntry {
	ls.mu.RLock()
	defer ls.mu.RUnlock()

	var filtered []LogEntry
	for i := len(ls.entries) - 1; i >= 0 && len(filtered) < n; i-- {
		if ls.entries[i].Level >= minLevel {
			filtered = append(filtered, ls.entries[i])
		}
	}

	// Reverse to chronological order
	for i, j := 0, len(filtered)-1; i < j; i, j = i+1, j-1 {
		filtered[i], filtered[j] = filtered[j], filtered[i]
	}
	return filtered
}

// GetStats returns summary statistics about logged entries.
func (ls *LogService) GetStats() map[string]interface{} {
	ls.mu.RLock()
	defer ls.mu.RUnlock()

	counts := map[string]int{
		"debug": 0,
		"info":  0,
		"warn":  0,
		"error": 0,
	}
	for _, e := range ls.entries {
		switch e.Level {
		case LogLevelDebug:
			counts["debug"]++
		case LogLevelInfo:
			counts["info"]++
		case LogLevelWarn:
			counts["warn"]++
		case LogLevelError:
			counts["error"]++
		}
	}

	return map[string]interface{}{
		"total_entries": len(ls.entries),
		"counts":        counts,
		"min_level":     ls.minLevel.String(),
		"capacity":      ls.maxSize,
	}
}

// Clear removes all stored log entries.
func (ls *LogService) Clear() {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	ls.entries = ls.entries[:0]
}

// SetMinLevel updates the minimum log level filter.
func (ls *LogService) SetMinLevel(level LogLevel) {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	ls.minLevel = level
}

// Formatted returns a formatted string of the entry for display.
func (e LogEntry) Formatted() string {
	base := fmt.Sprintf("[%s] %s [%s] %s",
		e.Timestamp.Format("2006-01-02 15:04:05.000"),
		e.LevelStr,
		e.Component,
		e.Message,
	)
	if len(e.Fields) > 0 {
		for k, v := range e.Fields {
			base += fmt.Sprintf(" %s=%s", k, v)
		}
	}
	return base
}
