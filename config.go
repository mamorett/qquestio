package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type Config struct {
	QdrantURL         string `json:"qdrant_url"`
	QdrantAPIKey      string `json:"qdrant_api_key"`
	EmbeddingURL      string `json:"embedding_url"`
	EmbeddingAPIKey   string `json:"embedding_api_key"`
	EmbeddingModel    string `json:"embedding_model"`
	OpenAIURL         string `json:"openai_url"`
	OpenAIAPIKey      string `json:"openai_api_key"`
	OpenAIModel       string `json:"openai_model"`
	DefaultCollection string `json:"default_collection"`
	RerankerURL       string `json:"reranker_url"`
	RerankerAPIKey    string `json:"reranker_api_key"`
	RerankerModel     string `json:"reranker_model"`
	SearchCap         int    `json:"search_cap,omitempty"`
	RerankerPool      int    `json:"reranker_pool,omitempty"`
	// ContextLimit is the estimated token budget for the conversation history.
	// When the conversation exceeds 85% of this limit, history is auto-compacted.
	// Default: 131072 (128k). Set to 0 to disable auto-compaction entirely.
	ContextLimit int `json:"context_limit,omitempty"`
}

func getConfigPath() string {
	home, err := os.UserHomeDir()
	if err == nil {
		return fmt.Sprintf("%s/.config/qquestio/config.json", home)
	}
	return "config.json"
}

// loadJSONConfig reads configuration from a config.json file if it exists.
func loadJSONConfig() (Config, bool) {
	var cfg Config
	configPath := getConfigPath()
	file, err := os.Open(configPath)
	if err != nil {
		if configPath != "config.json" {
			file, err = os.Open("config.json")
		}
	}
	if err != nil {
		return cfg, false
	}
	defer file.Close()

	err = json.NewDecoder(file).Decode(&cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: failed to parse config file: %v\n", err)
		return cfg, false
	}
	return cfg, true
}


func LoadConfig() (Config, error) {
	// 1. Start with values from config.json (if present)
	cfg, _ := loadJSONConfig()

	// 2. Override with system environment variables (highest precedence)
	overrideFromEnv := func(target *string, envKey string) {
		if val := os.Getenv(envKey); val != "" {
			*target = val
		}
	}

	overrideFromEnv(&cfg.QdrantURL, "QDRANT_URL")
	overrideFromEnv(&cfg.QdrantAPIKey, "QDRANT_API_KEY")
	overrideFromEnv(&cfg.EmbeddingURL, "EMBEDDING_URL")
	overrideFromEnv(&cfg.EmbeddingAPIKey, "EMBEDDING_API_KEY")
	overrideFromEnv(&cfg.EmbeddingModel, "EMBEDDING_MODEL")
	overrideFromEnv(&cfg.OpenAIURL, "OPENAI_URL")
	overrideFromEnv(&cfg.OpenAIAPIKey, "OPENAI_API_KEY")
	overrideFromEnv(&cfg.OpenAIModel, "OPENAI_MODEL")
	overrideFromEnv(&cfg.DefaultCollection, "DEFAULT_COLLECTION")
	overrideFromEnv(&cfg.RerankerURL, "RERANKER_URL")
	overrideFromEnv(&cfg.RerankerAPIKey, "RERANKER_API_KEY")
	overrideFromEnv(&cfg.RerankerModel, "RERANKER_MODEL")
	if val := os.Getenv("SEARCH_CAP"); val != "" {
		if n, err := strconv.Atoi(val); err == nil && n >= 0 {
			cfg.SearchCap = n
		}
	}
	if val := os.Getenv("RERANKER_POOL"); val != "" {
		if n, err := strconv.Atoi(val); err == nil && n >= 0 {
			cfg.RerankerPool = n
		}
	}
	if val := os.Getenv("CONTEXT_LIMIT"); val != "" {
		if n, err := strconv.Atoi(val); err == nil && n >= 0 {
			cfg.ContextLimit = n
		}
	}

	// Apply the default context limit when neither JSON nor env configured it.
	// CONTEXT_LIMIT=0 (or context_limit: 0 in JSON) explicitly disables auto-compact.
	if cfg.ContextLimit == 0 && os.Getenv("CONTEXT_LIMIT") == "" {
		cfg.ContextLimit = 131072
	}

	// 3. Validate and construct super-clear error message if variables are missing
	var missing []string
	if cfg.QdrantURL == "" {
		missing = append(missing, "QDRANT_URL")
	}
	if cfg.QdrantAPIKey == "" {
		missing = append(missing, "QDRANT_API_KEY")
	}
	if cfg.EmbeddingURL == "" {
		missing = append(missing, "EMBEDDING_URL")
	}
	if cfg.EmbeddingModel == "" {
		missing = append(missing, "EMBEDDING_MODEL")
	}
	if cfg.OpenAIURL == "" {
		missing = append(missing, "OPENAI_URL")
	}
	if cfg.OpenAIModel == "" {
		missing = append(missing, "OPENAI_MODEL")
	}
	if cfg.DefaultCollection == "" {
		missing = append(missing, "DEFAULT_COLLECTION")
	}

	if len(missing) > 0 {
		return Config{}, fmt.Errorf(
			"\n"+
				"========================================================================\n"+
				"                       QQUESTIO CONFIGURATION ERROR                     \n"+
				"========================================================================\n"+
				"Missing the following required configuration options:\n"+
				"  - %s\n\n"+
				"Please configure these using one of the following methods:\n\n"+
				"1. System Environment Variables:\n"+
				"   export QDRANT_URL=\"http://localhost:6333\"\n"+
				"   export QDRANT_API_KEY=\"your-key\"\n"+
				"   export EMBEDDING_URL=\"http://localhost:8080\"\n"+
				"   export EMBEDDING_API_KEY=\"optional-embedding-key\"\n"+
				"   export EMBEDDING_MODEL=\"nomic-embed\"\n"+
				"   export OPENAI_URL=\"http://localhost:4000\"\n"+
				"   export OPENAI_API_KEY=\"optional-openai-key\"\n"+
				"   export OPENAI_MODEL=\"llama3\"\n"+
				"   export DEFAULT_COLLECTION=\"documents\"\n"+
				"   export RERANKER_URL=\"http://localhost:8009\"   # optional\n"+
				"   export RERANKER_MODEL=\"reranker-model\"         # optional\n"+
				"   export SEARCH_CAP=\"0\"                          # optional, 0=full-corpus\n"+
				"   export RERANKER_POOL=\"0\"                       # optional, 0=auto\n"+
				"   export CONTEXT_LIMIT=\"131072\"                  # optional, 0=disable auto-compact\n\n"+
				"2. A \"config.json\" File in $HOME/.config/qquestio/config.json or the current directory:\n"+
				"   {\n"+
				"     \"qdrant_url\": \"http://localhost:6333\",\n"+
				"     \"qdrant_api_key\": \"your-key\",\n"+
				"     \"embedding_url\": \"http://localhost:8080\",\n"+
				"     \"embedding_api_key\": \"optional-embedding-key\",\n"+
				"     \"embedding_model\": \"nomic-embed\",\n"+
				"     \"openai_url\": \"http://localhost:4000\",\n"+
				"     \"openai_api_key\": \"optional-openai-key\",\n"+
				"     \"openai_model\": \"llama3\",\n"+
				"     \"default_collection\": \"documents\",\n"+
				"     \"reranker_url\": \"http://localhost:8009\",    ← optional\n"+
				"     \"reranker_model\": \"reranker-model\",         ← optional\n"+
				"     \"search_cap\": 0,                              ← optional, 0=full-corpus\n"+
				"     \"reranker_pool\": 0,                           ← optional, 0=auto\n"+
				"     \"context_limit\": 131072                       ← optional, 0=disable auto-compact\n"+
				"   }\n"+
				"========================================================================",
			strings.Join(missing, "\n  - "),
		)
	}

	return cfg, nil
}
