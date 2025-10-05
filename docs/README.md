# GPUX Documentation

This directory contains the source for GPUX documentation, built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

## 🚀 Quick Start

### View Documentation Locally

```bash
# Serve docs with live reload
uv run mkdocs serve

# Open browser to http://127.0.0.1:8000
```

### Build Documentation

```bash
# Build static site
uv run mkdocs build

# Output will be in site/ directory
```

## 📁 Structure

```
docs/
├── index.md                 # Landing page
├── tutorial/                # Step-by-step tutorials
├── guide/                   # User guides
├── examples/                # Real-world examples
├── advanced/                # Advanced topics
├── deployment/              # Deployment guides
├── platforms/               # Platform-specific guides
├── reference/               # API reference
├── about/                   # About & meta
├── stylesheets/             # Custom CSS
├── javascripts/             # Custom JavaScript
└── assets/                  # Images and static files
```

## ✍️ Writing Documentation

### Style Guide

- **Tone**: Friendly but professional
- **Voice**: Second person ("you")
- **Tense**: Present tense
- **Code**: Always include working examples
- **Length**: Keep pages focused (500-1500 words)

### Markdown Extensions

We use several extensions for enhanced formatting:

#### Admonitions

```markdown
!!! note "Optional Title"
    This is a note.

!!! tip
    This is a helpful tip.

!!! warning
    This is a warning.

!!! danger
    This is critical.
```

#### Code Blocks with Tabs

```markdown
=== "Python"
    ```python
    print("Hello")
    ```

=== "Bash"
    ```bash
    echo "Hello"
    ```
```

#### Code Annotations

```python
def example():
    x = 1  # (1)!
    return x

# (1) This is an annotation
```

### Adding a New Page

1. **Create the Markdown file** in the appropriate directory
2. **Add to navigation** in `mkdocs.yml`
3. **Preview changes** with `uv run mkdocs serve`
4. **Test build** with `uv run mkdocs build --strict`

## 🛠️ Development

### Install Dependencies

```bash
# Install documentation dependencies
uv sync --group docs
```

### Check for Errors

```bash
# Strict build (fails on warnings)
uv run mkdocs build --strict
```

### Preview Changes

```bash
# Start dev server with live reload
uv run mkdocs serve

# Custom port
uv run mkdocs serve --dev-addr 127.0.0.1:9000
```

## 🚢 Deployment

### GitHub Pages

```bash
# Deploy to gh-pages branch
uv run mkdocs gh-deploy
```

### Custom Domain

Update `mkdocs.yml`:

```yaml
site_url: https://docs.gpux.io
```

## 📝 TODO

See [DOCUMENTATION_PLAN.md](../DOCUMENTATION_PLAN.md) for the complete implementation plan.

### Phase 1: Foundation ✅
- [x] Setup MkDocs
- [x] Create directory structure
- [x] Write landing page
- [x] Write tutorial index
- [x] Write installation guide

### Phase 2: Tutorial (In Progress)
- [x] Installation
- [ ] First Steps
- [ ] Configuration
- [ ] Running Inference
- [ ] Benchmarking
- [ ] Serving

### Phase 3-8: To Do
- [ ] User Guide
- [ ] Examples
- [ ] Advanced Topics
- [ ] Deployment Guides
- [ ] Platform Guides
- [ ] API Reference

## 🆘 Help

- **MkDocs Documentation**: https://www.mkdocs.org/
- **Material Theme**: https://squidfunk.github.io/mkdocs-material/
- **Markdown Guide**: https://www.markdownguide.org/

## 📄 License

Documentation is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
