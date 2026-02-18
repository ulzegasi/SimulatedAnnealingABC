#!/usr/bin/env bash

env_name="sabc-dev"

# ## install env
micromamba create --file environment.${env_name:?}.yml --yes
# ## clean pip cache
micromamba run -n ${env_name} python -m pip cache purge
# ## clean micromamba cache
micromamba clean --all --force-pkgs-dirs --yes

# ## install npm packages
micromamba run -n ${env_name} npm install -g \
    opencode-ai \
    markdownlint-cli2 markdownlint-cli2-formatter-pretty markdownlint-cli2-formatter-summarize
# ## clean npm cache
micromamba run -n ${env_name} npm cache clean --force

# ## opencode setup (ensure auth.json is symlinked to the correct location)
if [ -f "${WORKSPACE_FOLDER:?}/.opencode/auth.json" ]; then
    mkdir -p ~/.local/share/opencode
    ln -s ${WORKSPACE_FOLDER:?}/.opencode/auth.json ~/.local/share/opencode/auth.json
else
    echo "Warning: ${WORKSPACE_FOLDER:?}/.opencode/auth.json not found. Please ensure you have the correct auth.json file for opencode and symlink it to ~/.local/share/opencode/auth.json"
fi