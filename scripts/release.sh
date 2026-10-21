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
#   2. the same files are pushed to the package's own GitHub repository,
#      with a tag of the same version.
#
# The GitHub repository and the hub therefore hold the same bytes at the
# same version, and the hub's hash is the identity of both.
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

echo "-- github"
REPO="$HOME/github/bend/$name"
if [ ! -d "$REPO/.git" ]; then
  echo "no checkout of the repository at $REPO" >&2
  exit 1
fi
# the package's files, and nothing else: the same set the hub holds
rm -rf "$REPO/.files"
mkdir -p "$REPO/.files"
for f in "$DIR"/*; do
  base="$(basename "$f")"
  case "$base" in
    tests) continue ;;
    bend.toml) continue ;;
  esac
  cp -r "$f" "$REPO/.files/"
done
cd "$REPO"
rm -rf -- *
mv .files/* . 2>/dev/null || true
rmdir .files 2>/dev/null || true
cp "$MANIFEST" ./bend.toml
git add -A
if git diff --cached --quiet; then
  echo "the repository is already at these files"
else
  git -c user.name="adeerkhan" -c user.email="adeerkhan@users.noreply.github.com" \
    commit -q -m "$name $version: the release of $name"
fi
git tag -f "v$version" -m "$name $version" >/dev/null
git push -q origin HEAD
git push -q -f origin "refs/tags/v$version"
echo "pushed $name $version to github and to the hub ($hash)"
