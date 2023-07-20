# Development Workflow

## Table of contents

- [Development Workflow](#development-workflow)
  - [Table of contents](#table-of-contents)
  - [Main branches](#main-branches)
  - [Development model](#development-model)
      - [Feature branches](#feature-branches)
      - [Release branches](#release-branches)
      - [Hotfix branches](#hotfix-branches)
    - [Git operations](#git-operations)
      - [Create a feature branch](#create-a-feature-branch)
      - [Merge feature branch into staging](#merge-feature-branch-into-staging)
      - [Create release branch](#create-release-branch)
      - [Finish a release branch](#finish-a-release-branch)
      - [Create the hotfix branch](#create-the-hotfix-branch)
      - [Finishing a hotfix branch](#finishing-a-hotfix-branch)
  - [TODO](#todo)

## Main branches

Bittensor is composed of TWO main branches, **master** and **staging**

**master**
- master Bittensor's live production branch. This branch should only be touched and merged into by the core develpment team. This branch is protected, but you should make no attempt to push or merge into it reguardless. 

**staging**
- staging is Bittensor's development branch. This branch is being continuously updated and merged into. This is the branch where you will propose and merge changes.

## Development model

#### Feature branches

- May branch off from: `staging`
- Must merge back into: `staging`
- Branch naming convention:
    - Anything except master, staging, finney, release/* or hotfix/*
    - Suggested: `feature/<ticket>/<descriptive-sentence>`

When implementing new features, hotfixes, bugfixes, or upgrades, it is wise to adhere to a strict naming and merging convention, whenever possible.

**Branch naming and merging convention:**


Feature branches are used to develop new features for the upcoming or a distant future release. When starting development of a feature, the target release in which this feature will be incorporated may well be unknown at that point. 

The essence of a feature branch is that it exists as long as the feature is in development, but will eventually be merged into `staging` (to definitely add the new feature to the upcoming release) or discarded (in case of a disappointing experiment).

Generally, you should try to minimize the lifespan of feature branches. As soon as you merge a feature into 'staging', you should immidiately delete the feature branch. This will be strictly enforced. Excess branches creates tech debt and confusion between development teams and parties.

#### Release branches

- Please branch off from: `staging`
- Please merge back into: `staging` then into: `master`
- Branch naming convention:
    - STRONGLY suggested format `release/5.1.0/descriptive-message/creator's-name`

Release branches support preparation of a new production release. Furthermore, they allow for minor bug fixes and preparing meta-data for a release (e.g.: version number, configuration, etc.). By doing all of this work on a release branch, the `staging` branch is cleared to receive features for the next big release.

This new branch may exist there for a while, until the release may be rolled out definitely. During that time, bug fixes may be applied in this branch, rather than on the `staging` branch. Adding large new features here is strictly prohibited. They must be merged into `staging`, and therefore, wait for the next big release.

#### Hotfix branches

- Please branch off from: `master` or `staging`
- Please merge back into: `staging` then into: `master`
- Branch naming convention:
    - REQUIRED format: `hotfix/3.3.4/descriptive-message/creator's-name` 

Hotfix branches are very much like release branches in that they are also meant to prepare for a new production release, albeit unplanned. They arise from the necessity to act immediately upon an undesired state of a live production version. When a critical bug in a production version must be resolved immediately, a hotfix branch may be branched off from the corresponding tag on the master branch that marks the production version.

The essence is that work of team members, on the `staging` branch, can continue, while another person is preparing a quick production fix.

### Git operations

#### Create a feature branch

1. Branch from the **staging** branch.
    1. Command: `git checkout -b feature/my-feature staging`

> Rebase frequently with the updated staging branch so you do not face big conflicts before submitting your pull request. Remember, syncing your changes with other developers could also help you avoid big conflicts.

#### Merge feature branch into staging

In other words, integrate your changes into a branch that will be tested and prepared for release.

- Switch branch to staging: `git checkout staging`
- Merging feature branch into staging: `git merge --no-ff feature/my-feature`
- Pushing changes to staging: `git push origin staging`
- Delete feature branch: `git branch -d feature/my-feature` (alternatively, this can be navigated on the GitHub web UI)

This operation is done by Github when merging a PR.

So, what you have to keep in mind is:
- Open the PR against the `staging` branch.
- After merging a PR you should delete your feature branch. This will be strictly enforced.

#### Create release branch

- Create branch from staging: `git checkout -b release/3.4.0/descriptive-message/creator's_name staging`
- Updating version with major or minor: `./scripts/update_version.sh major|minor`
- Commit file changes with new version: `git commit -a -m "Updated version to 3.4.0"`

#### Finish a release branch

In other words, releasing stable code and generating a new version for bittensor.

- Switch branch to master: `git checkout master`
- Merging release branch into master: `git merge --no-ff release/3.4.0/optional-descriptive-message`
- Tag changeset: `git tag -a v3.4.0 -m "Releasing v3.4.0: some comment about it"`
- Pushing changes to master: `git push origin master`
- Pushing tags to origin: `git push origin --tags`

To keep the changes made in the __release__ branch, we need to merge those back into `staging`:

- Switch branch to staging: `git checkout staging`.
- Merging release branch into staging: `git merge --no-ff release/3.4.0/optional-descriptive-message`

This step may well lead to a merge conflict (probably even, since we have changed the version number). If so, fix it and commit.

After this the release branch may be removed, since we don’t need it anymore:

- `git branch -d release/3.4.0/descriptive-message/creator's-name`

#### Create the hotfix branch

- Create branch from master:`git checkout -b hotfix/3.3.4/descriptive-message/creator's-name master`
- Update patch version: `./scripts/update_version.sh patch`
- Commit file changes with new version: `git commit -a -m "Updated version to 3.3.4"`

Then, fix the bug and commit the fix in one or more separate commits:
- `git commit -m "Fixed critical production issue"`

#### Finishing a hotfix branch

When finished, the bugfix needs to be merged back into `master`, but also needs to be merged back into `staging`, in order to safeguard that the bugfix is included in the next release as well. This is completely similar to how release branches are finished.

First, update master and tag the release.

- Switch branch to master: `git checkout master`
- Merge changes into master: `git merge --no-ff hotfix/3.3.4/optional-descriptive-message`
- Tag new version: `git tag -a v3.3.4 -m "Releasing v3.3.4: descriptive comment about the hotfix"`
- Pushing changes to master: `git push origin master`
- Pushing tags to origin: `git push origin --tags`

Next, include the bugfix in `staging`, too:

- Switch branch to staging: `git checkout staging`
- Merge changes into staging: `git merge --no-ff hotfix/3.3.4/descriptive-message/creator's-name`
- Pushing changes to origin/staging: `git push origin staging`

The one exception to the rule here is that, **when a release branch currently exists, the hotfix changes need to be merged into that release branch, instead of** `staging`. Back-merging the bugfix into the __release__ branch will eventually result in the bugfix being merged into `develop` too, when the release branch is finished. (If work in develop immediately requires this bugfix and cannot wait for the release branch to be finished, you may safely merge the bugfix into develop now already as well.)

Finally, we remove the temporary branch:

- `git branch -d hotfix/3.3.4/optional-descriptive-message`

## TODO

- Knowing if master and staging are different
- Knowing what is in staging that is not merge yet
    - Document with not released developments
    - When merged into staging, generate the information exposing what's merged into staging but not release.
    - When merged into master, generate github release and release notes.
- CircleCI job 
    - Merge staging into master and release version (needed to release code)
    - Build and Test Bittensor (needed to merge PRs)
