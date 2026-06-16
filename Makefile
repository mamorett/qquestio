# Makefile for QQuestio RAG TUI

BINARY_NAME=qquestio
BUILD_DIR=bin

.PHONY: all build clean test build-all build-linux-amd64 build-linux-arm64 build-windows-amd64 build-windows-arm64 build-darwin-amd64 build-darwin-arm64

all: build

build:
	go build -o $(BINARY_NAME)

test:
	go test -v ./...

clean:
	rm -rf $(BUILD_DIR) $(BINARY_NAME) debug.log

build-all: build-linux-amd64 build-linux-arm64 build-windows-amd64 build-windows-arm64 build-darwin-amd64 build-darwin-arm64

build-linux-amd64:
	GOOS=linux GOARCH=amd64 go build -o $(BUILD_DIR)/$(BINARY_NAME)-linux-amd64

build-linux-arm64:
	GOOS=linux GOARCH=arm64 go build -o $(BUILD_DIR)/$(BINARY_NAME)-linux-arm64

build-windows-amd64:
	GOOS=windows GOARCH=amd64 go build -o $(BUILD_DIR)/$(BINARY_NAME)-windows-amd64.exe

build-windows-arm64:
	GOOS=windows GOARCH=arm64 go build -o $(BUILD_DIR)/$(BINARY_NAME)-windows-arm64.exe

build-darwin-amd64:
	GOOS=darwin GOARCH=amd64 go build -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-amd64

build-darwin-arm64:
	GOOS=darwin GOARCH=arm64 go build -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-arm64
