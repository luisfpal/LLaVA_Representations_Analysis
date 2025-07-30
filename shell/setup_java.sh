#!/bin/bash

# Simple Java 8 setup for SPICE evaluation from pycocoevalcap.eval.COCOEvalCap
echo "🔧 Setting up Java 8 for SPICE evaluation..."

# Check if Java 8 is already installed
if [ -d "/usr/lib/jvm/java-8-openjdk-amd64" ]; then
    echo "✅ Java 8 already installed"
else
    echo "📦 Installing Java 8..."
    sudo apt update && sudo apt install -y openjdk-8-jdk
fi

# Set environment variables
export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"

echo "✅ Java setup complete!"
java -version 