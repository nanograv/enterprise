# enterprise Release Process

**Status**: Fully automated (GitHub release triggers PyPI upload)

## Prerequisites
- Maintainer access to [nanograv/enterprise](https://github.com/nanograv/enterprise)
- GitHub environment `deploy` configured with trusted publishing
- `id-token: write` permission enabled

## Release Steps

### 1. Prepare Release
```bash
# Ensure main branch is up to date
git checkout main
git pull origin main

# Run tests locally
make test
make lint

# Check that all CI tests are passing on GitHub
# Visit: https://github.com/nanograv/enterprise/actions
```

### 2. Create GitHub Release
1. Go to [enterprise releases](https://github.com/nanograv/enterprise/releases)
2. Click "Create a new release"
3. Choose a tag version (e.g., `v3.4.6`)
4. Set release title (e.g., `v3.4.6`)
5. Add release notes describing changes
6. Click "Publish release"

### 3. Automated Process
- GitHub Actions will automatically:
  - Run comprehensive tests (linting, pytest, codecov)
  - Build source distribution and wheel
  - Test deployability with `twine check`
  - Upload to PyPI using trusted publishing (no API tokens needed)

## Version Management
- Uses `setuptools_scm` - version automatically generated from git tags
- Current version visible in `enterprise/version.py` (auto-generated)
- No manual version file updates needed
- Version format: `X.Y.Z.devN+g<commit_hash>` for development versions

## Troubleshooting
- **Upload fails**: Check GitHub environment permissions and trusted publishing setup
- **Tests fail**: Ensure all dependencies are installed locally before release
- **Version issues**: Check git tags are properly formatted (e.g., `v3.4.6`)

## Release Checklist
- [ ] All tests passing on GitHub Actions
- [ ] Release notes prepared
- [ ] GitHub release created
- [ ] PyPI upload successful (check [PyPI page](https://pypi.org/project/enterprise-pulsar/))
- [ ] Documentation updated if needed

## Example Release Commands
```bash
# Tag a new version
git tag v3.4.6
git push origin v3.4.6

# Then create GitHub release via web interface
```

## Trusted Publishing
This package uses GitHub's trusted publishing feature, which means:
- No API tokens or passwords needed
- Uses GitHub's OIDC (OpenID Connect) for authentication
- More secure than traditional API key authentication
- Configured in GitHub repository settings under "Environments"
