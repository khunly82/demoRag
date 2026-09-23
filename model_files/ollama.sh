/bin/ollama serve &  # Start Ollama server in the background
pid=$!  # Capture process ID

ollama pull mistral:latest

ollama run mistral:latest

wait $pid