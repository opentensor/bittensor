# Development Workflow

## Table of contents

1. [Main branches](#main-branches)
1. [Development model](#development-model)
    1. [Supporting branches](#supporting-branches)
        1. [Feature branches](#feature-branches)
        1. [Release branches](#release-branches)
        1. [Hotfix branches](#hotfix-branches)
    1. [Git operations](#git-operations)
        1. [Create a feature branch](#create-a-feature-branch)
        1. [Merge feature branch into nobunaga](#merge-feature-branch-into-nobunaga)
        1. [Create release branch](#create-release-branch)
        1. [Finish a release branch](#finish-a-release-branch)
        1. [Create a hotfix branch](#create-a-hotfix-branch)
        1. [Finishing a hotfix branch](#finishing-a-hotfix-branch)

## Main branches

The repo holds two main branches with an infinite lifetime:
- master
- nobunaga

We consider `origin/master` to be the main branch where the source code of HEAD always reflects a **__production-ready__** state.

We consider `origin/nobunaga` to be the main branch where the source code of HEAD always reflects a state with the **__latest delivered development__** changes for the next release. Some would call this the `"integration branch"`. This is where any automatic nightly builds would be built from.

## Development model

### Supporting branches

Each of these branches have a specific purpose and are bound to strict rules as to which branches may be their originating branch and which branches must be their merge targets. We will walk through them in a minute

#### Feature branches

- May branch off from: `nobunaga`
- Must merge back into: `nobunaga`
- Branch naming convention:
    - Anything except master, nobunaga, finney, release/* or hotfix/*
    - Suggested: `feature/<ticket>/<descriptive-sentence>`

Feature branches are used to develop new features for the upcoming or a distant future release. When starting development of a feature, the target release in which this feature will be incorporated may well be unknown at that point. 

The essence of a feature branch is that it exists as long as the feature is in development, but will eventually be merged back into `nobunaga` (to definitely add the new feature to the upcoming release) or discarded (in case of a disappointing experiment).

#### Release branches

- May branch off from: `nobunaga`
- Must merge back into: `nobunaga` and `master`
- Branch naming convention:
    - Suggested format `release/3.4.0/optional-descriptive-message`

Release branches support preparation of a new production release. Furthermore, they allow for minor bug fixes and preparing meta-data for a release (e.g.: version number, configuration, etc.). By doing all of this work on a release branch, the `nobunaga` branch is cleared to receive features for the next big release.

This new branch may exist there for a while, until the release may be rolled out definitely. During that time, bug fixes may be applied in this branch, rather than on the `nobunaga` branch. Adding large new features here is strictly prohibited. They must be merged into `nobunaga`, and therefore, wait for the next big release.

#### Hotfix branches

- May branch off from: `master`
- Must merge back into: `nobunaga` and `master`
- Branch naming convention:
    - Suggested format: `hotfix/3.3.4/optional-descriptive-message` 

Hotfix branches are very much like release branches in that they are also meant to prepare for a new production release, albeit unplanned. They arise from the necessity to act immediately upon an undesired state of a live production version. When a critical bug in a production version must be resolved immediately, a hotfix branch may be branched off from the corresponding tag on the master branch that marks the production version.

The essence is that work of team members, on the `nobunaga` branch, can continue, while another person is preparing a quick production fix.

### Git operations

#### Create a feature branch

1. Branch from the **nobunaga** branch.
    1. Command: `git checkout -b feature/my-feature nobunaga`

> Try to rebase frequently with the updated nobunaga branch so you do not face big conflicts before submitting your pull request. Remember, syncing your changes with other developers could also help you avoid big conflicts.

#### Merge feature branch into nobunaga

In other words, integrate your changes into a branch that will be tested and prepared for release.

- Switch branch to nobunaga: `git checkout nobunaga`
- Merging feature branch into nobunaga: `git merge --no-ff feature/my-feature`
- Pushing changes to nobunaga: `git push origin nobunaga`
- Delete feature branch: `git branch -d feature/my-feature`

This operation is done by Github when merging a PR.

So, what you have to keep in mind is:
- Open the PR against the `nobunaga` branch.
- After merging a PR you just have to delete your feature branch.

#### Create release branch

- Create branch from nobunaga: `git checkout -b release/3.4.0/optional-descriptive-message nobunaga`
- Updating version with major or minor: `./scripts/update_version.sh major|minor`
- Commit file changes with new version: `git commit -a -m "Updated version to 3.4.0"`

#### Finish a release branch

In other words, releasing stable code and generating a new version for bittensor.

- Switch branch to master: `git checkout master`
- Merging release branch into master: `git merge --no-ff release/3.4.0/optional-descriptive-message`
- Tag changeset: `git tag -a v3.4.0 -m "Releasing v3.4.0: some comment about it"`
- Pushing changes to master: `git push origin master`
- Pushing tags to origin: `git push origin --tags`

To keep the changes made in the __release__ branch, we need to merge those back into `nobunaga`:

- Switch branch to nobunaga: `git checkout nobunaga`.
- Merging release branch into nobunaga: `git merge --no-ff release/3.4.0/optional-descriptive-message`

This step may well lead to a merge conflict (probably even, since we have changed the version number). If so, fix it and commit.

After this the release branch may be removed, since we don’t need it anymore:

- `git branch -d release/3.4.0/optional-descriptive-message`

#### Create the hotfix branch

- Create branch from master:`git checkout -b hotfix/3.3.4/optional-descriptive-message master`
- Update patch version: `./scripts/update_version.sh patch`
- Commit file changes with new version: `git commit -a -m "Updated version to 3.3.4"`

Then, fix the bug and commit the fix in one or more separate commits:
- `git commit -m "Fixed critical production issue"`

#### Finishing a hotfix branch

When finished, the bugfix needs to be merged back into `master`, but also needs to be merged back into `nobunaga`, in order to safeguard that the bugfix is included in the next release as well. This is completely similar to how release branches are finished.

First, update master and tag the release.

- Switch branch to master: `git checkout master`
- Merge changes into master: `git merge --no-ff hotfix/3.3.4/optional-descriptive-message`
- Tag new version: `git tag -a v3.3.4 -m "Releasing v3.3.4: some comment about the hotfix"`
- Pushing changes to master: `git push origin master`
- Pushing tags to origin: `git push origin --tags`

Next, include the bugfix in `nobunaga`, too:

- Switch branch to nobunaga: `git checkout nobunaga`
- Merge changes into nobunaga: `git merge --no-ff hotfix/3.3.4/optional-descriptive-message`
- Pushing changes to origin/nobunaga: `git push origin nobunaga`

The one exception to the rule here is that, **when a release branch currently exists, the hotfix changes need to be merged into that release branch, instead of** `nobunaga`. Back-merging the bugfix into the __release__ branch will eventually result in the bugfix being merged into `develop` too, when the release branch is finished. (If work in develop immediately requires this bugfix and cannot wait for the release branch to be finished, you may safely merge the bugfix into develop now already as well.)

Finally, we remove the temporary branch:

- `git branch -d hotfix/3.3.4/optional-descriptive-message`

## TODO

- Changing the name of the develop branch from nobunaga to `integration`
    - Because sometimes nobunaga are going to have a release branch.
- Knowing if master and nobunaga are different
- Knowing what is in nobunaga that is not merge yet
    - Document with not released developments
    - When merged into nobunaga, generate the information exposing what's merged into nobunaga but not release.
    - When merged into master, generate github release and release notes.
- CircleCI job 
    - Merge nobunaga into master and release version (needed to release code)
    - Build and Test bittensor (needed to merge PRs)