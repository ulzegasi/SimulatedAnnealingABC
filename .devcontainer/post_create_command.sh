#!/usr/bin/env bash



# ## install mkdocs env
micromamba create --file environment.sabc-dev.yml --yes
# ## clean pip cache
micromamba run -n mkdocs python -m pip cache purge
# ## clean micromamba cache
micromamba clean --all --force-pkgs-dirs --yes