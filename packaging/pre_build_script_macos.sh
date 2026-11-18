set -euxo pipefail

pip install cmake ninja
echo "/home/runner/.local/bin" >> "$GITHUB_PATH"
echo $PATH
