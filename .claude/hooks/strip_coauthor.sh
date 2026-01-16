#!/bin/bash
# Claude Code PreToolUse hook to strip Co-Authored-By lines from git commits
# This hook intercepts Bash tool calls and modifies git commit commands

# Read JSON input from stdin
input=$(cat)

# Extract tool name and command
tool_name=$(echo "$input" | jq -r '.tool_name // empty')
command=$(echo "$input" | jq -r '.tool_input.command // empty')

# Only process Bash tool calls with git commit
if [[ "$tool_name" == "Bash" ]] && [[ "$command" == *"git commit"* ]]; then
    # Strip Co-Authored-By lines mentioning Claude/Anthropic
    new_command=$(echo "$command" | sed -E 's/Co-Authored-By:[^\n]*[Cc]laude[^\n]*\n?//g; s/Co-Authored-By:[^\n]*anthropic[^\n]*\n?//gi')

    # Also strip "Generated with Claude Code" lines
    new_command=$(echo "$new_command" | sed -E 's/.*Generated with \[Claude Code\].*\n?//g; s/.*claude\.com.*\n?//gi')

    # If command was modified, output the updated input
    if [[ "$new_command" != "$command" ]]; then
        echo "$input" | jq --arg cmd "$new_command" '.tool_input.command = $cmd | {updatedInput: .tool_input}'
        exit 0
    fi
fi

# Allow the command to proceed unchanged
exit 0
