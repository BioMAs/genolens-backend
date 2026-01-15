#!/bin/bash
# Script to initialize Ollama in Docker with BioMistral model
# Run this after starting docker-compose for the first time

set -e

echo "🤖 Initializing Ollama with BioMistral model..."

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama service to start..."
until docker exec backend-ollama-1 ollama list &> /dev/null; do
    sleep 2
done

echo "✅ Ollama is ready!"

# Check if biomistral is already installed
if docker exec backend-ollama-1 ollama list | grep -q "biomistral"; then
    echo "✅ BioMistral model already installed"
else
    echo "📥 Downloading BioMistral model (4.1 GB)..."
    echo "⏱️  This may take 5-10 minutes depending on your connection..."
    docker exec backend-ollama-1 ollama pull biomistral
    echo "✅ BioMistral model installed successfully!"
fi

# Optional: Pull other models
echo ""
echo "📦 Optional: Install additional models?"
echo "  - llama3.1:8b (4.7 GB, general purpose, fast)"
echo "  - mixtral:8x7b (26 GB, high quality, slower)"
echo ""
read -p "Install llama3.1:8b? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Downloading Llama 3.1 8B..."
    docker exec backend-ollama-1 ollama pull llama3.1:8b
    echo "✅ Llama 3.1 8B installed!"
fi

echo ""
echo "🎉 Ollama setup complete!"
echo ""
echo "📊 Installed models:"
docker exec backend-ollama-1 ollama list
echo ""
echo "🚀 You can now use AI interpretation in GenoLens!"
echo "   API endpoint: http://localhost:8001/datasets/ai/status"
