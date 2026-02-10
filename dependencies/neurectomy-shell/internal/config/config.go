package config

import (
	"os"

	"gopkg.in/yaml.v3"
)

// Config holds the server configuration.
type Config struct {
	Port    int      `yaml:"port"`
	DataDir string   `yaml:"data_dir"`
	Vault   VaultCfg `yaml:"vault"`
	TEE     TEECfg   `yaml:"tee"`
	Audit   AuditCfg `yaml:"audit"`
}

// VaultCfg configures ΣVAULT integration.
type VaultCfg struct {
	Endpoint string `yaml:"endpoint"`
	KeyID    string `yaml:"key_id"`
	Enabled  bool   `yaml:"enabled"`
}

// TEECfg configures the Trusted Execution Environment.
type TEECfg struct {
	Provider     string `yaml:"provider"` // "sev-snp", "tdx", "simulate"
	Attestation  bool   `yaml:"attestation"`
	MemorySizeMB int    `yaml:"memory_size_mb"`
}

// AuditCfg configures the immutable audit logger.
type AuditCfg struct {
	LogPath   string `yaml:"log_path"`
	Encrypted bool   `yaml:"encrypted"`
}

// Load reads configuration from a YAML file.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}

	return &cfg, nil
}

// Default returns the default configuration.
func Default() *Config {
	return &Config{
		Port:    8443,
		DataDir: "/var/lib/neurectomy",
		Vault: VaultCfg{
			Endpoint: "localhost:9090",
			Enabled:  false,
		},
		TEE: TEECfg{
			Provider:     "simulate",
			Attestation:  false,
			MemorySizeMB: 4096,
		},
		Audit: AuditCfg{
			LogPath:   "/var/log/neurectomy/audit.log",
			Encrypted: true,
		},
	}
}
