package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

type Config struct {
	ActiveConfigName     string `json:"active_config_name,omitempty"`
	QdrantURL            string `json:"qdrant_url"`
	QdrantAPIKey         string `json:"qdrant_api_key"`
	QdrantVectorName     string `json:"qdrant_vector_name,omitempty"`
	EmbeddingURL         string `json:"embedding_url"`
	EmbeddingAPIKey      string `json:"embedding_api_key"`
	EmbeddingModel       string `json:"embedding_model"`
	OpenAIURL            string `json:"openai_url"`
	OpenAIAPIKey         string `json:"openai_api_key"`
	OpenAIModel          string `json:"openai_model"`
	DefaultCollection    string `json:"default_collection"`
	RerankerURL          string `json:"reranker_url"`
	RerankerAPIKey       string `json:"reranker_api_key"`
	RerankerModel        string `json:"reranker_model"`
	SearchCap            int    `json:"search_cap,omitempty"`
	RerankerPool         int    `json:"reranker_pool,omitempty"`
	HTTPTimeoutSeconds   int    `json:"http_timeout_seconds,omitempty"`
	SkillsRequireConfirm bool   `json:"skills_require_confirm,omitempty"`
	// ContextLimit is the estimated token budget for the conversation history.
	// Default: 131072 (128k). When the conversation exceeds 85% of this limit,
	// history is auto-compacted. Set to 0 to disable auto-compaction entirely.
	ContextLimit    int    `json:"context_limit,omitempty"`
	QueryRewrite    string `json:"query_rewrite,omitempty"` // "llm", "heuristic", or "off"
	OpenAIMaxTokens int    `json:"openai_max_tokens,omitempty"`
}

type configFile struct {
	Configurations       map[string]json.RawMessage `json:"configurations"`
	DefaultConfiguration string                     `json:"default_configuration"`
	Config
}

func getConfigPath() string {
	home, err := os.UserHomeDir()
	if err == nil {
		return fmt.Sprintf("%s/.config/qquestio/config.json", home)
	}
	return "config.json"
}

func defaultConfig() Config {
	return Config{
		HTTPTimeoutSeconds: 60,
		ContextLimit:       131072,
		QueryRewrite:       "llm",
	}
}

// loadJSONConfigFile reads configuration from a config.json file if it exists.
func loadJSONConfigFile() (configFile, bool) {
	file, _, ok := loadJSONConfigFileWithPath()
	return file, ok
}

// loadJSONConfigFileWithPath reads configuration and returns the actual path used.
func loadJSONConfigFileWithPath() (configFile, string, bool) {
	file := configFile{
		Config: defaultConfig(),
	}
	configPath := getConfigPath()
	actualPath := configPath
	f, err := os.Open(configPath)
	if err != nil {
		if configPath != "config.json" {
			f, err = os.Open("config.json")
			if err == nil {
				actualPath = "config.json"
			}
		}
	}
	if err != nil {
		return file, "", false
	}
	defer f.Close()

	err = json.NewDecoder(f).Decode(&file)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: failed to parse config file: %v\n", err)
		return file, actualPath, false
	}
	return file, actualPath, true
}

func LoadConfig(configName ...string) (Config, error) {
	// 1. Start with default values (overridden by config.json if present)
	cfg := defaultConfig()
	var activeName string

	file, ok := loadJSONConfigFile()
	if ok {
		var selectedName string
		if len(configName) > 0 && configName[0] != "" {
			selectedName = configName[0]
		} else if file.DefaultConfiguration != "" {
			selectedName = file.DefaultConfiguration
		}

		if selectedName != "" {
			raw, found := file.Configurations[selectedName]
			if !found {
				// Get list of valid configuration names for a better error message
				var names []string
				for k := range file.Configurations {
					names = append(names, k)
				}
				sort.Strings(names)
				if len(names) > 0 {
					return Config{}, fmt.Errorf("configuration %q not found. Available configurations: %s", selectedName, strings.Join(names, ", "))
				}
				return Config{}, fmt.Errorf("configuration %q not found (no configurations defined in file)", selectedName)
			}

			// Start with root config
			cfg = file.Config
			// Unmarshal named configuration on top of it to override fields
			if err := json.Unmarshal(raw, &cfg); err != nil {
				return Config{}, fmt.Errorf("failed to parse configuration %q: %w", selectedName, err)
			}
			activeName = selectedName
		} else {
			cfg = file.Config
		}
	}

	cfg.ActiveConfigName = activeName

	// 2. Override with system environment variables (highest precedence)
	overrideFromEnv := func(target *string, envKey string) {
		if val := os.Getenv(envKey); val != "" {
			*target = val
		}
	}

	overrideFromEnv(&cfg.QdrantURL, "QDRANT_URL")
	overrideFromEnv(&cfg.QdrantAPIKey, "QDRANT_API_KEY")
	overrideFromEnv(&cfg.QdrantVectorName, "QDRANT_VECTOR_NAME")
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

	overrideIntFromEnv := func(target *int, envKey string) {
		if val := os.Getenv(envKey); val != "" {
			n, err := strconv.Atoi(val)
			if err != nil || n < 0 {
				fmt.Fprintf(os.Stderr, "Warning: invalid integer value %q for %s, ignoring\n", val, envKey)
				return
			}
			*target = n
		}
	}
	overrideIntFromEnv(&cfg.SearchCap, "SEARCH_CAP")
	overrideIntFromEnv(&cfg.RerankerPool, "RERANKER_POOL")
	overrideIntFromEnv(&cfg.ContextLimit, "CONTEXT_LIMIT")
	overrideIntFromEnv(&cfg.HTTPTimeoutSeconds, "QQUESTIO_HTTP_TIMEOUT")
	overrideIntFromEnv(&cfg.OpenAIMaxTokens, "OPENAI_MAX_TOKENS")

	// Apply CLI overrides if explicitly set (highest precedence)
	if cliSearchCap >= 0 {
		cfg.SearchCap = cliSearchCap
	}
	if envVal := os.Getenv("QQUESTIO_SKILLS_REQUIRE_CONFIRM"); envVal != "" {
		if envVal == "1" || strings.EqualFold(envVal, "true") {
			cfg.SkillsRequireConfirm = true
		} else if envVal == "0" || strings.EqualFold(envVal, "false") {
			cfg.SkillsRequireConfirm = false
		}
	}
	if cliSafe {
		cfg.SkillsRequireConfirm = true
	}
	if val := os.Getenv("QUERY_REWRITE"); val != "" {
		cfg.QueryRewrite = val
	}

	// Apply defaults
	if cfg.HTTPTimeoutSeconds <= 0 {
		cfg.HTTPTimeoutSeconds = 60
	}
	if cfg.QueryRewrite == "" {
		cfg.QueryRewrite = "llm"
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
				"   export CONTEXT_LIMIT=\"131072\"                  # optional, 0=disable auto-compact\n"+
				"   export QQUESTIO_HTTP_TIMEOUT=\"60\"              # optional, default 60s\n\n"+
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
				"     \"context_limit\": 131072,                      ← optional, 0=disable auto-compact\n"+
				"     \"http_timeout_seconds\": 60                    ← optional, default 60s\n"+
				"   }\n"+
				"========================================================================",
			strings.Join(missing, "\n  - "),
		)
	}

	return cfg, nil
}

func GetAvailableConfigs() ([]string, string, error) {
	file, ok := loadJSONConfigFile()
	if !ok {
		return nil, "", fmt.Errorf("could not load config file")
	}
	var names []string
	for k := range file.Configurations {
		names = append(names, k)
	}
	sort.Strings(names)
	return names, file.DefaultConfiguration, nil
}

// maskSecret masks API keys or secrets for secure display.
func maskSecret(secret string) string {
	if secret == "" {
		return lipgloss.NewStyle().Foreground(nord3).Italic(true).Render("(none)")
	}
	if len(secret) <= 6 {
		return "••••••"
	}
	return secret[:3] + "•••" + secret[len(secret)-2:]
}

// resolveProfileConfig overlays a named configuration on top of the root configuration.
func resolveProfileConfig(file configFile, profileName string) Config {
	cfg := file.Config
	if raw, found := file.Configurations[profileName]; found {
		_ = json.Unmarshal(raw, &cfg)
	}
	cfg.ActiveConfigName = profileName
	return cfg
}

func renderProfileCard(name string, cfg Config, isDefault bool) string {
	labelStyle := lipgloss.NewStyle().Foreground(nord9).Bold(true).Width(17)
	valStyle := lipgloss.NewStyle().Foreground(nord6)
	mutedStyle := lipgloss.NewStyle().Foreground(nord3)
	accentStyle := lipgloss.NewStyle().Foreground(nord8)

	var rows []string

	// Collection
	col := cfg.DefaultCollection
	if col == "" {
		col = mutedStyle.Render("(none)")
	} else {
		col = lipgloss.NewStyle().Foreground(nord15).Bold(true).Render(col)
	}
	rows = append(rows, fmt.Sprintf("%s %s", labelStyle.Render("Collection:"), col))

	// Qdrant
	qdrantVal := cfg.QdrantURL
	if qdrantVal == "" {
		qdrantVal = mutedStyle.Render("(none)")
	}
	if cfg.QdrantVectorName != "" {
		qdrantVal += fmt.Sprintf(" %s", mutedStyle.Render("(vector: "+cfg.QdrantVectorName+")"))
	}
	if cfg.QdrantAPIKey != "" {
		qdrantVal += fmt.Sprintf(" %s", mutedStyle.Render("[key: "+maskSecret(cfg.QdrantAPIKey)+"]"))
	}
	rows = append(rows, fmt.Sprintf("%s %s", labelStyle.Render("Qdrant URL:"), qdrantVal))

	// Embedding
	embedVal := cfg.EmbeddingURL
	if embedVal == "" {
		embedVal = mutedStyle.Render("(none)")
	}
	if cfg.EmbeddingModel != "" {
		embedVal += fmt.Sprintf(" %s", accentStyle.Render("["+cfg.EmbeddingModel+"]"))
	}
	if cfg.EmbeddingAPIKey != "" {
		embedVal += fmt.Sprintf(" %s", mutedStyle.Render("[key: "+maskSecret(cfg.EmbeddingAPIKey)+"]"))
	}
	rows = append(rows, fmt.Sprintf("%s %s", labelStyle.Render("Embedding:"), embedVal))

	// OpenAI / LLM
	llmVal := cfg.OpenAIURL
	if llmVal == "" {
		llmVal = mutedStyle.Render("(none)")
	}
	if cfg.OpenAIModel != "" {
		llmVal += fmt.Sprintf(" %s", accentStyle.Render("["+cfg.OpenAIModel+"]"))
	}
	if cfg.OpenAIMaxTokens > 0 {
		llmVal += fmt.Sprintf(" %s", mutedStyle.Render(fmt.Sprintf("(max tokens: %s)", formatNumber(cfg.OpenAIMaxTokens))))
	}
	if cfg.OpenAIAPIKey != "" {
		llmVal += fmt.Sprintf(" %s", mutedStyle.Render("[key: "+maskSecret(cfg.OpenAIAPIKey)+"]"))
	}
	rows = append(rows, fmt.Sprintf("%s %s", labelStyle.Render("OpenAI / LLM:"), llmVal))

	// Reranker
	rerankVal := "(disabled / none)"
	if cfg.RerankerURL != "" {
		rerankVal = cfg.RerankerURL
		if cfg.RerankerModel != "" {
			rerankVal += fmt.Sprintf(" %s", accentStyle.Render("["+cfg.RerankerModel+"]"))
		}
		poolStr := "auto"
		if cfg.RerankerPool > 0 {
			poolStr = fmt.Sprintf("%d", cfg.RerankerPool)
		}
		rerankVal += fmt.Sprintf(" %s", mutedStyle.Render("(pool: "+poolStr+")"))
		if cfg.RerankerAPIKey != "" {
			rerankVal += fmt.Sprintf(" %s", mutedStyle.Render("[key: "+maskSecret(cfg.RerankerAPIKey)+"]"))
		}
	} else {
		rerankVal = mutedStyle.Render(rerankVal)
	}
	rows = append(rows, fmt.Sprintf("%s %s", labelStyle.Render("Reranker:"), rerankVal))

	// Search Cap & Context Limit
	capStr := "0 (full corpus)"
	if cfg.SearchCap > 0 {
		capStr = fmt.Sprintf("%s candidates", formatNumber(cfg.SearchCap))
	}
	ctxStr := "0 (disabled)"
	if cfg.ContextLimit > 0 {
		ctxStr = fmt.Sprintf("%s tokens", formatNumber(cfg.ContextLimit))
	}
	rows = append(rows, fmt.Sprintf("%s %s", labelStyle.Render("Search Cap:"), valStyle.Render(capStr)))
	rows = append(rows, fmt.Sprintf("%s %s", labelStyle.Render("Context Limit:"), valStyle.Render(ctxStr)))

	// Options
	safeStr := "false"
	if cfg.SkillsRequireConfirm {
		safeStr = lipgloss.NewStyle().Foreground(nord13).Render("true (confirm required)")
	}
	timeoutStr := fmt.Sprintf("%ds", cfg.HTTPTimeoutSeconds)
	if cfg.HTTPTimeoutSeconds <= 0 {
		timeoutStr = "60s"
	}
	qRewrite := cfg.QueryRewrite
	if qRewrite == "" {
		qRewrite = "llm"
	}
	rows = append(rows, fmt.Sprintf("%s %s   %s %s   %s %s",
		labelStyle.Width(15).Render("Query Rewrite:"), valStyle.Render(qRewrite),
		lipgloss.NewStyle().Foreground(nord9).Bold(true).Render("Timeout:"), valStyle.Render(timeoutStr),
		lipgloss.NewStyle().Foreground(nord9).Bold(true).Render("Skills Safe:"), safeStr,
	))

	borderColor := nord3
	nameBadge := lipgloss.NewStyle().Bold(true).Foreground(nord8).Render(name)
	if isDefault {
		borderColor = nord14
		nameBadge = fmt.Sprintf("%s %s",
			lipgloss.NewStyle().Bold(true).Foreground(nord14).Render(name),
			lipgloss.NewStyle().Background(nord14).Foreground(nord0).Bold(true).Padding(0, 1).Render("DEFAULT"),
		)
	}

	cardStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(borderColor).
		Padding(0, 1)

	headerLine := fmt.Sprintf(" Profile: %s", nameBadge)
	return fmt.Sprintf("%s\n%s", headerLine, cardStyle.Render(strings.Join(rows, "\n")))
}

// FormatConfigurations returns a beautifully styled string listing all configurations.
func FormatConfigurations() string {
	file, path, ok := loadJSONConfigFileWithPath()

	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(nord6).
		Background(nord1).
		Padding(0, 2).
		Render("QQuestio Configurations")

	var b strings.Builder
	b.WriteString("\n" + title + "\n\n")

	if !ok {
		warnStyle := lipgloss.NewStyle().Foreground(nord13)
		mutedStyle := lipgloss.NewStyle().Foreground(nord3)
		b.WriteString(warnStyle.Render("  ⚠ No configuration file found.") + "\n")
		b.WriteString(mutedStyle.Render(fmt.Sprintf("    Checked: %s and ./config.json", getConfigPath())) + "\n\n")
		b.WriteString(lipgloss.NewStyle().Foreground(nord4).Render("  To configure QQuestio, create a config.json file or set environment variables.") + "\n")
		return b.String()
	}

	infoKeyStyle := lipgloss.NewStyle().Foreground(nord9).Bold(true).Width(18)
	infoValStyle := lipgloss.NewStyle().Foreground(nord4)

	b.WriteString(fmt.Sprintf("  %s %s\n", infoKeyStyle.Render("Config file:"), infoValStyle.Render(path)))
	defaultName := file.DefaultConfiguration
	if defaultName == "" {
		defaultName = "(none specified)"
	}
	b.WriteString(fmt.Sprintf("  %s %s\n", infoKeyStyle.Render("Default profile:"), lipgloss.NewStyle().Foreground(nord14).Bold(true).Render(defaultName)))

	var profileNames []string
	for k := range file.Configurations {
		profileNames = append(profileNames, k)
	}
	sort.Strings(profileNames)

	if len(profileNames) > 0 {
		for _, name := range profileNames {
			cfg := resolveProfileConfig(file, name)
			isDefault := (name == file.DefaultConfiguration)
			b.WriteString("\n" + renderProfileCard(name, cfg, isDefault) + "\n")
		}
		b.WriteString("\n" + renderProfileCard("base (root template)", file.Config, false) + "\n")
	} else {
		b.WriteString("\n" + renderProfileCard("default (root)", file.Config, true) + "\n")
	}

	return b.String()
}
