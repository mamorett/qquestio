package main

import (
	"os"
	"testing"
)

func TestLoadConfig_Missing(t *testing.T) {
	os.Clearenv()
	_ = os.Remove("config.json") // Clean up any config.json that might exist

	cfg, err := LoadConfig()
	if err == nil {
		t.Fatalf("expected error when configuration options are missing, got nil (config: %+v)", cfg)
	}

	errMsg := err.Error()
	if !contains(errMsg, "QQUESTIO CONFIGURATION ERROR") {
		t.Errorf("expected error message to contain helper header, got:\n%s", errMsg)
	}
	if !contains(errMsg, "config.json") {
		t.Errorf("expected error message to explain JSON config usage, got:\n%s", errMsg)
	}
}

func TestLoadConfig_ValidOpenAI(t *testing.T) {
	os.Clearenv()
	t.Setenv("QDRANT_URL", "http://localhost:6333")
	t.Setenv("QDRANT_API_KEY", "test-key")
	t.Setenv("EMBEDDING_URL", "http://localhost:8080")
	t.Setenv("EMBEDDING_API_KEY", "test-embedding-key")
	t.Setenv("EMBEDDING_MODEL", "nomic-embed")
	t.Setenv("OPENAI_URL", "http://localhost:4000")
	t.Setenv("OPENAI_API_KEY", "test-openai-key")
	t.Setenv("OPENAI_MODEL", "llama3")
	t.Setenv("DEFAULT_COLLECTION", "documents")

	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if cfg.OpenAIURL != "http://localhost:4000" {
		t.Errorf("expected OpenAIURL 'http://localhost:4000', got %s", cfg.OpenAIURL)
	}
	if cfg.OpenAIAPIKey != "test-openai-key" {
		t.Errorf("expected OpenAIAPIKey 'test-openai-key', got %s", cfg.OpenAIAPIKey)
	}
	if cfg.OpenAIModel != "llama3" {
		t.Errorf("expected OpenAIModel 'llama3', got %s", cfg.OpenAIModel)
	}
	if cfg.EmbeddingAPIKey != "test-embedding-key" {
		t.Errorf("expected EmbeddingAPIKey 'test-embedding-key', got %s", cfg.EmbeddingAPIKey)
	}
}

func TestLoadConfig_JSONConfig(t *testing.T) {
	os.Clearenv()

	jsonContent := `{
		"qdrant_url": "http://localhost:6333",
		"qdrant_api_key": "json-key",
		"embedding_url": "http://localhost:8080",
		"embedding_api_key": "json-api-key",
		"embedding_model": "nomic-embed",
		"openai_url": "http://localhost:4600",
		"openai_api_key": "json-api-key",
		"openai_model": "llama3-json",
		"default_collection": "documents"
	}`

	err := os.WriteFile("config.json", []byte(jsonContent), 0644)
	if err != nil {
		t.Fatalf("failed to write mock config.json file: %v", err)
	}
	defer os.Remove("config.json")

	// 1. Should load values from JSON
	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.QdrantAPIKey != "json-key" {
		t.Errorf("expected QANT_API_KEY 'json-key' from JSON, got %s", cfg.QdrantAPIKey)
	}
	if cfg.OpenAIURL != "http://localhost:4600" {
		t.Errorf("expected OpenAIURL 'http://localhost:4600', got %s", cfg.OpenAIURL)
	}
	if cfg.OpenAIAPIKey != "json-api-key" {
		t.Errorf("expected OpenAIAPIKey 'json-api-key' from JSON, got %s", cfg.OpenAIAPIKey)
	}

	// 2. Precedence test: system env variable overrides config.json
	t.Setenv("QDRANT_API_KEY", "system-overridden")
	cfg, err = LoadConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.QdrantAPIKey != "system-overridden" {
		t.Errorf("expected QDRANT_API_KEY to be overridden by system env, got %s", cfg.QdrantAPIKey)
	}
}

// helper contains check
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 || (s[0:len(substr)] == substr || stringsContains(s, substr)))
}

func stringsContains(s, substr string) bool {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
