#!/usr/bin/env bash
# Copy the repository to DAS-6 over ssh as a tar stream (no rsync needed, works from Git Bash).
#
# Authentication uses an ssh-agent that already holds the passphrase-protected key. In Git Bash,
# once per session:   eval "$(ssh-agent -a ~/.ssh/agent.sock)" && ssh-add ~/.ssh/id_rsa
# (ControlMaster does not work on Windows, so every call here opens its own connection through the agent.)
#
#   scripts/sync_to_das6.sh
#
# Environment: DAS6_HOST (default skalra@fs2.das6.science.uva.nl), DAS6_REPO_DIR (default
# /home/skalra/Elastoformer-MOO). Files are overwritten; nothing on the remote is deleted.
# .git, local paper copies and caches are not sent.
set -euo pipefail
HOST="${DAS6_HOST:-skalra@fs2.das6.science.uva.nl}"
REMOTE_DIR="${DAS6_REPO_DIR:-/home/skalra/Elastoformer-MOO}"
export SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-$HOME/.ssh/agent.sock}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[[ -S "$SSH_AUTH_SOCK" ]] || { echo "ERROR: no ssh-agent socket at $SSH_AUTH_SOCK (see the header of this script)" >&2; exit 4; }
ssh-add -l >/dev/null 2>&1 || { echo "ERROR: the ssh-agent holds no key; run: ssh-add ~/.ssh/id_rsa" >&2; exit 4; }

# The Hugging Face token (~/.hf_token on this machine) is copied to ~/.hf_token on the cluster over stdin, so it
# never appears on a command line (visible in `ps`) and the remote file is created owner-only.
if [[ -f "$HOME/.hf_token" ]]; then
  ssh -o BatchMode=yes "$HOST" 'umask 077; cat > "$HOME/.hf_token"; chmod 600 "$HOME/.hf_token"' < "$HOME/.hf_token"
  echo "copied ~/.hf_token to $HOST (owner-only)" >&2
else
  echo "note: no ~/.hf_token on this machine; gated models will not be downloadable on the cluster" >&2
fi

echo "syncing $(pwd) -> $HOST:$REMOTE_DIR" >&2
tar czf - --exclude=.git --exclude='related papers' --exclude=__pycache__ --exclude=.pytest_cache \
  --exclude='*.pdf' --exclude='*.aux' --exclude='*.log' . \
  | ssh -o BatchMode=yes "$HOST" \
      "mkdir -p '$REMOTE_DIR' && tar xzf - -C '$REMOTE_DIR' && chmod +x '$REMOTE_DIR'/*.sh '$REMOTE_DIR'/scripts/*.sh && echo synced \$(find '$REMOTE_DIR' -type f | wc -l) files"
