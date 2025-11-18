# Use git subtree directly (no submodule at all)

In your repo:

```bash
cd /path/to/your-repo

git remote add foo-origin git@github.com:someone/foo.git

# Add /bar into your repo under bar/ (one-time)
git subtree add --prefix=bar foo-origin main --squash # adjust branch name if needed
```

Later, to update:

```bash
git subtree pull --prefix=bar foo-origin main --squash
```

```bash
Layout:
your-repo/
bar/ # just normal tracked files, not a submodule
...
```
