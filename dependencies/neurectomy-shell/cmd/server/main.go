package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/iamthegreatdestroyer/ryzanstein/dependencies/neurectomy-shell/internal/config"
	"github.com/iamthegreatdestroyer/ryzanstein/dependencies/neurectomy-shell/internal/server"
)

func main() {
	configPath := flag.String("config", "config.yaml", "Path to configuration file")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Printf("Warning: config load failed (%v), using defaults", err)
		cfg = config.Default()
	}

	srv, err := server.New(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create server: %v\n", err)
		os.Exit(1)
	}

	log.Printf("neurectomy-shell server starting on :%d", cfg.Port)
	if err := srv.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
		os.Exit(1)
	}
}
