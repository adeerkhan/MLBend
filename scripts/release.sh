#!/usr/bin/env bash
# Release one package: to GitHub and to the hub, at one version.
#
#   scripts/release.sh <package directory>
#
# The package's directory holds its bend.toml, which names it and gives
# its version. The release is:
#
#   1. the package's files are published to the hub, and the name and
#      the version are linked to the hash the hub gives;
#   2. the release is committed here, in this one repository, and tagged
#      <name>-v<version>; the branch and the tag are pushed to origin.
#
# There is no repository per package: every package is a directory of this
# one. The hub's hash is the identity of the published bytes, and the tag
# is the point in this repository's history at which they went out.
#
# The hub's names are four numbers and never move: a change is a new
# version, and a published version is public and permanent.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE="${1:-}"
if [ -z "$PACKAGE" ]; then
  echo "usage: scripts/release.sh <package directory>" >&2
  exit 1
fi
DIR="$ROOT/$PACKAGE"
MANIFEST="$DIR/bend.toml"
if [ ! -f "$MANIFEST" ]; then
  echo "no bend.toml in $DIR" >&2
  exit 1
fi

# the name and the version, out of the package's own manifest
name="$(grep -E '^name *=' "$MANIFEST" | head -1 | cut -d'"' -f2)"
version="$(grep -E '^version *=' "$MANIFEST" | head -1 | cut -d'"' -f2)"
if [ -z "$name" ] || [ -z "$version" ]; then
  echo "the manifest of $PACKAGE has no name or no version" >&2
  exit 1
fi

BENDENV="${BENDENV:-$HOME/.bend/bin/bendenv}"
BEND="${BEND:-$HOME/.bend/bin/bend}"
if [ ! -x "$BENDENV" ]; then
  echo "no bendenv at $BENDENV: see https://github.com/adeerkhan/bend-env" >&2
  exit 1
fi

echo "== $name $version"

# a release is a commit of this repository: fix_versions.py below writes the
# sibling versions, so the tree must hold nothing else when it runs
cd "$ROOT"
if [ -n "$(git status --porcelain)" ]; then
  echo "the working tree is not clean: commit or stash before a release" >&2
  git status --short >&2
  exit 1
fi

echo "-- the sibling versions"
# a sibling that moved takes its import with it, or the program reads the
# published sibling rather than the one beside it
python3 "$ROOT/scripts/fix_versions.py" --write

echo "-- the hub"
cd "$DIR"
out="$("$BENDENV" publish main.bend 2>&1)"
echo "$out"
hash="$(echo "$out" | grep -oE '0x[0-9a-f]{32}' | tail -1)"
if [ -z "$hash" ]; then
  echo "the hub did not answer with a hash" >&2
  exit 1
fi
if [ "${SKIP_LINK:-0}" != "1" ]; then
  # the name may already hold this version: the hub's versions never
  # move, so the same hash under the same version is the same package
  if ! out="$("$BEND" link "$name@$version" "$hash" 2>&1)"; then
    have="$(curl -s "https://hub.bend-lang.com/name/$name" | grep -oE '"hash": ?"0x[0-9a-f]{32}"' | head -1 | grep -oE '0x[0-9a-f]{32}')"
    if [ "$have" = "$hash" ]; then
      echo "$name@$version already holds this hash"
    else
      echo "$out" >&2
      exit 1
    fi
  else
    echo "$out"
  fi
fi

echo "-- git"
cd "$ROOT"
tag="$name-v$version"
# the sibling versions fix_versions.py rewrote, and the package, are the
# release: the tree was clean when the publish started, so this is all of it
git add -A
if git diff --cached --quiet; then
  echo "this version is already committed"
else
  git -c user.name="adeerkhan" -c user.email="adeerkhan@users.noreply.github.com" \
    commit -q -m "$name $version: the release of $name"
fi
git tag -f "$tag" -m "$name $version" >/dev/null
git push -q origin HEAD
git push -q -f origin "refs/tags/$tag"
echo "$name $version is on the hub ($hash) and tagged $tag on GitHub"
