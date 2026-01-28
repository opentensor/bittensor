# Release Guidelines

The release manager will need the right permissions for:
  - github.com (includes the PyPI credentials)

If you are new in this role, ask for the proper setup you need to run this process manually.

## Process of release

1. Begin to draft a new release in [Github](https://github.com/opentensor/bittensor/releases/new), using the appropriate version tag
   1. Note we follow [semver](https://semver.org/)
2. Create a new branch off of staging, named `changelog/<VERSION>`
3. After generating the release notes, copy these into the [CHANGELOG.md](../CHANGELOG.md) file, with the appropriate header
4. Bump the version in [pyproject.toml](../pyproject.toml)
5. Open a Pull Request against staging for this changelog
6. Once approved and merged into staging, delete the branch, and create a new branch off staging, named `release/<VERSION>`
7. Push this branch, and open a PR against master, which should include the changelog from step 3
8. Once this passes tests, is approved, and merged to master, run [Build and Publish Python Package](https://github.com/opentensor/bittensor/actions/workflows/release.yml) with the new version
9. Verify the release is successful and pushed to [PyPI](https://pypi.org/project/bittensor/#history)
