# Release Guidelines

The release manager in charge can release a Bittensor version using two scripts:
  - [./scripts/release/versioning.sh](./scripts/release/versioning.sh)
  - [./scripts/release/release.sh](./scripts/release/release.sh)

The release manager will need the right permissions for:
  - github.com
  - pypi.org
  - hub.docker.com

If you are new in this role, ask for the proper setup you need to run this process manually.

## Process of release

1. Create a branch called `release/VERSION`, having VERSION with the version to release.
1. Within the release branch; Update the version using the versioning script:
  1. `./scripts/release/versioning.sh --update UPDATE_TYPE`, UPDATE_TYPE could be major, minor or patch.
1. Test the release branch and verify that it meets the requirements.
1. After merging the release branch; Run the release script

## Versioning script usage

Options:
  - -U, --update: type of update. It could be major, minor, patch or rc (release candidate).
  - -A, --apply: This specify to apply the release. Without this the versioning will just show a dry run with no changes.

## Release script usage

Options:
  - -A, --apply: This specify to apply the release. Without this the release will just show a dry run with no changes.
  - -T,--github-token: A github personal access token to interact with the Github API.

### Github token

Since you need to use a secret when releasing bittensor (github personal access token), I encourage you to use [pass](https://www.passwordstore.org/) or a similar tool that allows you to store the secret safely and not expose it in the history of the machine you use.

So you can have:
```
GITHUB_ACCESS_TOKEN=$(pass github/your_personal_token_with_permisions)
```

or
```
GITHUB_ACCESS_TOKEN=$(whatever you need to get the token safely)
```

### Executions

So, executing the script to release a minor version will be:

```
# For a dry run
./scripts/release/release.sh --version minor --github-token $GITHUB_ACCESS_TOKEN`
```

```
# Applying changes
./scripts/release/release.sh --apply --version minor --github-token $GITHUB_ACCESS_TOKEN`
```

## Checking release

After the execution of the release script we would have generated:
  - A new git tag in [github.com](https://github.com/opentensor/bittensor/tags)
  - A new github release in [github.com](https://github.com/opentensor/bittensor/releases)
  - A new pip package in [pypi.org](https://pypi.org/project/bittensor/#history)
  - A new docker image in [hub.docker.com](https://hub.docker.com/r/opentensorfdn/bittensor/tags)

## After release

After a Bittensor release we have to
- Update [cubit](https://github.com/opentensor/cubit).

### Updating cubit

1. Updating the [Dockerfile](https://github.com/opentensor/cubit/blob/master/docker/Dockerfile)
1. Building its docker image (follow its README instructions)
1. Push it to hub.docker.com
  1. The generated name will be the same but with `-cubit` in its name