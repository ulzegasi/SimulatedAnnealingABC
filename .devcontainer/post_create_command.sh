#!/usr/bin/env bash

# Create environment (includes npm packages from Makefile)
make env-dev

# Clean caches
make clean-cache

# Opencode setup (ensure auth.json is symlinked to the correct location)
if [ -f "${WORKSPACE_FOLDER:?}/.opencode/auth.json" ]; then
    mkdir -p ~/.local/share/opencode
    ln -s ${WORKSPACE_FOLDER:?}/.opencode/auth.json ~/.local/share/opencode/auth.json
else
    echo "Warning: ${WORKSPACE_FOLDER:?}/.opencode/auth.json not found."
    echo " Either do the setup for opencode manually or add the auth.json file to the .opencode directory in the workspace, e.g. by symlinking it on your local system."
fi
