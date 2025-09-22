#!/bin/bash

mkdir -p ~/.streamlit

# Only config.toml is needed
cat <<EOF > ~/.streamlit/config.toml
[server]
headless = true
enableCORS = false
port = $PORT
EOF
