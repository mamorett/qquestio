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

	// 3. OpenAIMaxTokens & ContextLimit defaults validation
	// OpenAIMaxTokens should be 0 when undefined (no limit)
	if cfg.OpenAIMaxTokens != 0 {
		t.Errorf("expected default OpenAIMaxTokens to be 0, got %d", cfg.OpenAIMaxTokens)
	}
	// ContextLimit should default to 131072 (128k) when not explicitly set
	if cfg.ContextLimit != 131072 {
		t.Errorf("expected default ContextLimit to be 131072, got %d", cfg.ContextLimit)
	}

	// 4. Honoring exact values check
	t.Setenv("OPENAI_MAX_TOKENS", "16384")
	t.Setenv("CONTEXT_LIMIT", "32768")
	cfg, err = LoadConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.OpenAIMaxTokens != 16384 {
		t.Errorf("expected OpenAIMaxTokens to be exactly 16384, got %d", cfg.OpenAIMaxTokens)
	}
	if cfg.ContextLimit != 32768 {
		t.Errorf("expected ContextLimit to be exactly 32768, got %d", cfg.ContextLimit)
	}
}

func TestLoadConfig_RerankerPool(t *testing.T) {
	// 1. Env override test
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
	t.Setenv("RERANKER_POOL", "75")

	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.RerankerPool != 75 {
		t.Errorf("expected RerankerPool 75, got %d", cfg.RerankerPool)
	}

	// 2. JSON config test
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
		"default_collection": "documents",
		"reranker_pool": 150
	}`

	err = os.WriteFile("config.json", []byte(jsonContent), 0644)
	if err != nil {
		t.Fatalf("failed to write mock config.json file: %v", err)
	}
	defer os.Remove("config.json")

	cfg, err = LoadConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.RerankerPool != 150 {
		t.Errorf("expected RerankerPool 150, got %d", cfg.RerankerPool)
	}
}

func TestLoadConfig_MultipleProfiles(t *testing.T) {
	os.Clearenv()

	jsonContent := `{
		"qdrant_url": "http://localhost:6333",
		"qdrant_api_key": "root-key",
		"embedding_url": "http://localhost:8080",
		"embedding_model": "nomic-embed",
		"openai_url": "http://localhost:4000",
		"openai_model": "llama3",
		"default_collection": "documents",
		"default_configuration": "dev",
		"configurations": {
			"dev": {
				"default_collection": "dev-docs",
				"openai_model": "llama3-dev"
			},
			"prod": {
				"qdrant_url": "http://prod-qdrant:6333",
				"default_collection": "prod-docs",
				"openai_model": "gpt-4",
				"qdrant_vector_name": "prod-vec"
			}
		}
	}`

	err := os.WriteFile("config.json", []byte(jsonContent), 0644)
	if err != nil {
		t.Fatalf("failed to write mock config.json file: %v", err)
	}
	defer os.Remove("config.json")

	// 1. Loading with no arguments should fall back to default_configuration ("dev")
	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.ActiveConfigName != "dev" {
		t.Errorf("expected ActiveConfigName to be 'dev', got %s", cfg.ActiveConfigName)
	}
	if cfg.DefaultCollection != "dev-docs" {
		t.Errorf("expected DefaultCollection to be 'dev-docs' from 'dev' profile, got %s", cfg.DefaultCollection)
	}
	if cfg.OpenAIModel != "llama3-dev" {
		t.Errorf("expected OpenAIModel to be 'llama3-dev' from 'dev' profile, got %s", cfg.OpenAIModel)
	}
	if cfg.QdrantURL != "http://localhost:6333" {
		t.Errorf("expected QdrantURL to inherit root 'http://localhost:6333', got %s", cfg.QdrantURL)
	}

	// 2. Loading explicit "prod" configuration
	cfgProd, err := LoadConfig("prod")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfgProd.ActiveConfigName != "prod" {
		t.Errorf("expected ActiveConfigName to be 'prod', got %s", cfgProd.ActiveConfigName)
	}
	if cfgProd.DefaultCollection != "prod-docs" {
		t.Errorf("expected DefaultCollection to be 'prod-docs', got %s", cfgProd.DefaultCollection)
	}
	if cfgProd.OpenAIModel != "gpt-4" {
		t.Errorf("expected OpenAIModel to be 'gpt-4', got %s", cfgProd.OpenAIModel)
	}
	if cfgProd.QdrantURL != "http://prod-qdrant:6333" {
		t.Errorf("expected QdrantURL to override to 'http://prod-qdrant:6333', got %s", cfgProd.QdrantURL)
	}
	if cfgProd.QdrantVectorName != "prod-vec" {
		t.Errorf("expected QdrantVectorName to be 'prod-vec', got %s", cfgProd.QdrantVectorName)
	}

	// 3. Loading non-existent configuration should error
	_, err = LoadConfig("staging")
	if err == nil {
		t.Fatal("expected error when loading non-existent profile, got nil")
	}
	if !contains(err.Error(), "configuration \"staging\" not found") {
		t.Errorf("expected error to mention staging not found, got: %v", err)
	}

	// 4. Test GetAvailableConfigs
	names, defaultConf, err := GetAvailableConfigs()
	if err != nil {
		t.Fatalf("unexpected error getting configs: %v", err)
	}
	if defaultConf != "dev" {
		t.Errorf("expected default config to be 'dev', got %s", defaultConf)
	}
	if len(names) != 2 || names[0] != "dev" || names[1] != "prod" {
		t.Errorf("expected names to be ['dev', 'prod'], got %v", names)
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
