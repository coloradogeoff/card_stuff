#!/bin/zsh

KEY_FILE="/Users/geoff/code/card_stuff/.openai-api-key.txt"
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "$KEY_FILE" ]; then
  OPENAI_API_KEY="$(tr -d '\r\n' < "$KEY_FILE")"
  export OPENAI_API_KEY
fi

/Users/geoff/opt/anaconda3/bin/python /Users/geoff/code/card_stuff/Card\ Namer\ App/card_namer.py
