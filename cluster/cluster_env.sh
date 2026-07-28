#!/bin/bash
# Source this file after cluster/tunnel_ontomem.sh is connected.

export OPENAI_API_KEY=${OPENAI_API_KEY:-ontomem-cluster}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:18000/v1}
export OPENAI_MODEL=${OPENAI_MODEL:-ontomem-llm}
export OPENAI_EMBED_API_KEY=${OPENAI_EMBED_API_KEY:-ontomem-cluster}
export OPENAI_EMBED_BASE_URL=${OPENAI_EMBED_BASE_URL:-http://127.0.0.1:18001/v1}
export OPENAI_EMBED_MODEL=${OPENAI_EMBED_MODEL:-ontomem-embed}
