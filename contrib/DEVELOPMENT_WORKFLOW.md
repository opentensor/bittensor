# Bittensor Development Workflow

## Table of contents

- [Bittensor Development Workflow](#bittensor-development-workflow)
  - [Main Branches](#main-branches)
  - [Development Model](#development-model)
      - [Feature Branches](#feature-branches)
      - [Release Branches](#release-branches)
      - [Hotfix Branches](#hotfix-branches)
  - [Git Operations](#git-operations)
      - [Creating a Feature Branch](#creating-a-feature-branch)
      - [Merging Feature Branch into Staging](#merging-feature-branch-into-staging)
      - [Creating a Release Branch](#creating-a-release-branch)
      - [Finishing a Release Branch](#finishing-a-release-branch)
      - [Creating a Hotfix Branch](#creating-a-hotfix-branch)
      - [Finishing a Hotfix Branch](#finishing-a-hotfix-branch)
  - [Continuous Integration (CI) and Continuous Deployment (CD)](#continuous-integration-ci-and-continuous-deployment-cd)
  - [Versioning and Release Notes](#versioning-and-release-notes)
  - [Pending Tasks](#pending-tasks)

## Main Branches

Bittensor's codebase consists of two main branches: **master** and **staging**.

**master**
- This is Bittensor's live production branch, which should only be updated by the core development team. This branch is protected, so refrain from pushing or merging into it unless authorized.

**staging**
- This branch is continuously updated and is where you propose and merge changes. It's essentially Bittensor's active development branch.

## Development Model

### Feature Branches

- Branch off from: `staging`
- Merge back into: `staging`
- Naming convention: `feature/<ticket>/<descriptive-sentence>`

Feature branches are used to develop new features for upcoming or future releases. They exist as long as the feature is in development, but will eventually be merged into `staging` or discarded. Always delete your feature branch after merging to avoid unnecessary clutter.

### Release Branches

- Branch off from: `staging`
- Merge back into: `staging` and then `master`
- Naming convention: `release/<version>/<descriptive-message>/<creator's-name>`

Release branches support the preparation of a new production release, allowing for minor bug fixes and preparation of metadata (version number, configuration, etc). All new features should be merged into `staging` and wait for the next big release.

### Hotfix Branches

General workflow:

- Branch off from: `master` or `staging`
- Merge back into: `staging` then `master`
- Naming convention: `hotfix/<version>/<descriptive-message>/<creator's-name>` 

Hotfix branches are meant for quick fixes in the production environment. When a critical bug in a production version must be resolved immediately, a hotfix branch is created.

## Git Operations

#### Create a feature branch

1. Branch from the **staging** branch.
    1. Command: `git checkout -b feature/my-feature staging`

> Rebase frequently with the updated staging branch so you do not face big conflicts before submitting your pull request. Remember, syncing your changes with other developers could also help you avoid big conflicts.

#### Merge feature branch into staging

In other words, integrate your changes into a branch that will be tested and prepared for release.

1. Switch branch to staging: `git checkout staging`
2. Merging feature branch into staging: `git merge --no-ff feature/my-feature`
3. Pushing changes to staging: `git push origin staging`
4. Delete feature branch: `git branch -d feature/my-feature` (alternatively, this can be navigated on the GitHub web UI)

This operation is done by Github when merging a PR.

So, what you have to keep in mind is:
- Open the PR against the `staging` branch.
- After merging a PR you should delete your feature branch. This will be strictly enforced.

#### Creating a release branch

1. Create branch from staging: `git checkout -b release/3.4.0/descriptive-message/creator's_name staging`
2. Updating version with major or minor: `./scripts/update_version.sh major|minor`
3. Commit file changes with new version: `git commit -a -m "Updated version to 3.4.0"`


#### Finishing a Release Branch

This involves releasing stable code and generating a new version for bittensor.

1. Switch branch to master: `git checkout master`
2. Merge release branch into master: `git merge --no-ff release/3.4.0/optional-descriptive-message`
3. Tag changeset: `git tag -a v3.4.0 -m "Releasing v3.4.0: some comment about it"`
4. Push changes to master: `git push origin master`
5. Push tags to origin: `git push origin --tags`

To keep the changes made in the __release__ branch, we need to merge those back into `staging`:

- Switch branch to staging: `git checkout staging`.
- Merging release branch into staging: `git merge --no-ff release/3.4.0/optional-descriptive-message`

This step may well lead to a merge conflict (probably even, since we have changed the version number). If so, fix it and commit.


#### Creating a hotfix branch
1. Create branch from master: `git checkout -b hotfix/3.3.4/descriptive-message/creator's-name master`
2. Update patch version: `./scripts/update_version.sh patch`
3. Commit file changes with new version: `git commit -a -m "Updated version to 3.3.4"`
4. Fix the bug and commit the fix: `git commit -m "Fixed critical production issue X"`

#### Finishing a Hotfix Branch

Finishing a hotfix branch involves merging the bugfix into both `master` and `staging`.

1. Switch branch to master: `git checkout master`
2. Merge hotfix into master: `git merge --no-ff hotfix/3.3.4/optional-descriptive-message`
3. Tag new version: `git tag -a v3.3.4 -m "Releasing v3.3.4: descriptive comment about the hotfix"`
4. Push changes to master: `git push origin master`
5. Push tags to origin: `git push origin --tags`
6. Switch branch to staging: `git checkout staging`
7. Merge hotfix into staging: `git merge --no-ff hotfix/3.3.4/descriptive-message/creator's-name`
8. Push changes to origin/staging: `git push origin staging`
9. Delete hotfix branch: `git branch -d hotfix/3.3.4/optional-descriptive-message`

The one exception to the rule here is that, **when a release branch currently exists, the hotfix changes need to be merged into that release branch, instead of** `staging`. Back-merging the bugfix into the __release__ branch will eventually result in the bugfix being merged into `develop` too, when the release branch is finished. (If work in develop immediately requires this bugfix and cannot wait for the release branch to be finished, you may safely merge the bugfix into develop now already as well.)

Finally, we remove the temporary branch:

- `git branch -d hotfix/3.3.4/optional-descriptive-message`
## Continuous Integration (CI) and Continuous Deployment (CD)

Continuous Integration (CI) is a software development practice where members of a team integrate their work frequently. Each integration is verified by an automated build and test process to detect integration errors as quickly as possible. 

Continuous Deployment (CD) is a software engineering approach in which software functionalities are delivered frequently through automated deployments.

- **CircleCI job**: Create jobs in CircleCI to automate the merging of staging into master and release version (needed to release code) and building and testing Bittensor (needed to merge PRs).

## Versioning and Release Notes

Semantic versioning helps keep track of the different versions of the software. When code is merged into master, generate a new version. 

Release notes provide documentation for each version released to the users, highlighting the new features, improvements, and bug fixes. When merged into master, generate GitHub release and release notes.

## Pending Tasks

- Determine if master and staging are different
- Determine what is in staging that is not merged yet
    - Document not released developments
    - When merged into staging, generate information about what's merged into staging but not released.
    - When merged into master, generate GitHub release and release notes.
- CircleCI jobs 
    - Merge staging into master and release version (needed to release code)
    - Build and Test Bittensor (needed to merge PRs)

This document can be improved as the Bittensor project continues to develop and change.
