# Contributing to Bittensor

The following is a set of guidelines for contributing to Bittensor, which are hosted in the [Opentensor Organization](https://github.com/opentensor) on GitHub. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table Of Contents
1. [I don't want to read this whole thing, I just have a question!!!](#i-dont-want-to-read-this-whole-thing-i-just-have-a-question)
1. [What should I know before I get started?](#what-should-i-know-before-i-get-started)
1. [Getting Started](#getting-started)
   1. [Good First Issue Label](#good-first-issue-label)
   1. [Beginner and Help-wanted Issues Label](#beginner-and-help-wanted-issues-label)
1. [How Can I Contribute?](#how-can-i-contribute)
   1. [Code Contribution General Guideline](#code-contribution-general-guidelines)
   1. [Pull Request Philosophy](#pull-request-philosophy)
   1. [Pull Request Process](#pull-request-process)
   1. [Testing](#testing)
   1. [Addressing Feedback](#addressing-feedback)
   1. [Squashing Commits](#squashing-commits)
   1. [Refactoring](#refactoring)
   1. [Peer Review](#peer-review)
 1. [Reporting Bugs](#reporting-bugs)
 1. [Suggesting Features](#suggesting-enhancements-and-features)


## I don't want to read this whole thing I just have a question!

> **Note:** Please don't file an issue to ask a question. You'll get faster results by using the resources below.

We have an official Discord server where the community chimes in with helpful advice if you have questions. 
This is the fastest way to get an answer and the core development team is active on Discord.

* [Official Bittensor Discord](https://discord.gg/7wvFuPJZgq)

## What should I know before I get started?
Bittensor is still in the Alpha stages, and as such you will likely run into some problems in deploying your model or installing Bittensor itself. If you run into an issue or end up resolving an issue yourself, feel free to create a pull request with a fix or with a fix to the documentation. The documentation repository can be found [here](https://github.com/opentensor/docs). 

Additionally, note that the core implementation of Bittensor consists of two separate repositories: [The core Bittensor code](https://github.com/opentensor/bittensor) and the Bittensor Blockchain [subtensor](https://github.com/opentensor/subtensor).

Supplemental repository for the Bittensor subnet template can be found [here](https://github.com/opentensor/bittensor-subnet-template). This is a great first place to look for getting your hands dirty and start learning and building on Bittensor. See the subnet links [page](https://github.com/opentensor/bittensor-subnet-template/blob/main/subnet_links.json) for a list of all the repositories for the active registered subnets.

## Getting Started
New contributors are very welcome and needed.
Reviewing and testing is highly valued and the most effective way you can contribute as a new contributor. It also will teach you much more about the code and process than opening pull requests. 

Before you start contributing, familiarize yourself with the Bittensor Core build system and tests. Refer to the documentation in the repository on how to build Bittensor core and how to run the unit tests, functional tests.

There are many open issues of varying difficulty waiting to be fixed. If you're looking for somewhere to start contributing, check out the [good first issue](https://github.com/opentensor/bittensor/labels/good%20first%20issue) list or changes that are up for grabs. Some of them might no longer be applicable. So if you are interested, but unsure, you might want to leave a comment on the issue first. Also peruse the [issues](https://github.com/opentensor/bittensor/issues) tab for all open issues.

### Good First Issue Label
The purpose of the good first issue label is to highlight which issues are suitable for a new contributor without a deep understanding of the codebase.

However, good first issues can be solved by anyone. If they remain unsolved for a longer time, a frequent contributor might address them.

You do not need to request permission to start working on an issue. However, you are encouraged to leave a comment if you are planning to work on it. This will help other contributors monitor which issues are actively being addressed and is also an effective way to request assistance if and when you need it.

### Beginner and Help-wanted Issues Label
You can start by looking through these `beginner` and `help-wanted` issues:

* [Beginner issues](https://github.com/opentensor/bittensor/labels/beginner) - issues which should only require a few lines of code, and a test or two.
* [Help wanted issues](https://github.com/opentensor/bittensor/labels/help%20wanted) - issues which should be a bit more involved than `beginner` issues.

## Communication Channels
Most communication about Bittensor development happens on Discord channel.
Here's the link of Discord community.
[Bittensor Discord](https://discord.com/channels/799672011265015819/799672011814862902)

And also here.
[Bittensor Community Discord](https://discord.com/channels/1120750674595024897/1120799375703162950)

## How Can I Contribute?

You can contribute to Bittensor in one of two main ways (as well as many others):
1. [Bug](#reporting-bugs) reporting and fixes
2. New features and Bittensor [suggesting-enhancements](#suggesting-enhancements)

> Please follow the Bittensor [style guide](./STYLE.md) regardless of your contribution type. 

Here is a high-level summary:
- Code consistency is crucial; adhere to established programming language conventions.
- Use `ruff format .` to format your Python code; it ensures readability and consistency.
- Write concise Git commit messages; summarize changes in ~50 characters.
- Follow these six commit rules:
  - Atomic Commits: Focus on one task or fix per commit.
  - Subject and Body Separation: Use a blank line to separate the subject from the body.
  - Subject Line Length: Keep it under 50 characters for readability.
  - Imperative Mood: Write subject line as if giving a command or instruction.
  - Body Text Width: Wrap text manually at 72 characters.
  - Body Content: Explain what changed and why, not how.
- Make use of your commit messages to simplify project understanding and maintenance.

> For clear examples of each of the commit rules, see the style guide's [rules](./STYLE.md#the-six-rules-of-a-great-commit) section.

### Code Contribution General Guidelines

> Review the Bittensor [style guide](./STYLE.md) and [development workflow](./DEVELOPMENT_WORKFLOW.md) before contributing. 

If you're looking to contribute to Bittensor but unsure where to start, please join our community [discord](https://discord.gg/bittensor), a developer-friendly Bittensor town square. Start with [#development](https://discord.com/channels/799672011265015819/799678806159392768) and [#bounties](https://discord.com/channels/799672011265015819/1095684873810890883) to see what issues are currently posted. For a greater understanding of Bittensor's usage and development, check the [Bittensor Documentation](https://bittensor.com/docs).

#### Pull Request Philosophy

Patchsets and enhancements should always be focused. A pull request could add a feature, fix a bug, or refactor code, but it should not contain a mixture of these. Please also avoid 'super' pull requests which attempt to do too much, are overly large, or overly complex as this makes review difficult. 

Specifically, pull requests must adhere to the following criteria:
- **Must** branch off from `staging`. Make sure that all your PRs are using `staging` branch as a base or will be closed.
- Contain fewer than 50 files. PRs with more than 50 files will be closed.
- Use the specific [template](./.github/pull_request_template.md) appropriate to your contribution.
- If a PR introduces a new feature, it *must* include corresponding tests.
- Other PRs (bug fixes, refactoring, etc.) should ideally also have tests, as they provide proof of concept and prevent regression.
- Categorize your PR properly by using GitHub labels. This aids in the review process by informing reviewers about the type of change at a glance.
- Make sure your code includes adequate comments. These should explain why certain decisions were made and how your changes work.
- If your changes are extensive, consider breaking your PR into smaller, related PRs. This makes your contributions easier to understand and review.
- Be active in the discussion about your PR. Respond promptly to comments and questions to help reviewers understand your changes and speed up the acceptance process.

Generally, all pull requests must:

  - Have a clear use case, fix a demonstrable bug or serve the greater good of the project (e.g. refactoring for modularisation).
  - Be well peer-reviewed.
  - Follow code style guidelines.
  - Not break the existing test suite.
  - Where bugs are fixed, where possible, there should be unit tests demonstrating the bug and also proving the fix.
  - Change relevant comments and documentation when behaviour of code changes.

#### Pull Request Process

Please follow these steps to have your contribution considered by the maintainers:

*Before* creating the PR:
1. Read the [development workflow](./DEVELOPMENT_WORKFLOW.md) defined for this repository to understand our workflow.
2. Ensure your PR meets the criteria stated in the 'Pull Request Philosophy' section.
3. Include relevant tests for any fixed bugs or new features as stated in the [testing guide](./TESTING.md).
4. Follow all instructions in [the template](https://github.com/opentensor/bittensor/blob/master/.github/PULL_REQUEST_TEMPLATE/pull_request_template.md) to create the PR.
5. Ensure your commit messages are clear and concise. Include the issue number if applicable.
6. If you have multiple commits, rebase them into a single commit using `git rebase -i`.
7. Explain what your changes do and why you think they should be merged in the PR description consistent with the [style guide](./STYLE.md).

*After* creating the PR:
1. Verify that all [status checks](https://help.github.com/articles/about-status-checks/) are passing after you submit your pull request. 
2. Label your PR using GitHub's labeling feature. The labels help categorize the PR and streamline the review process.
3. Document your code with comments that provide a clear understanding of your changes. Explain any non-obvious parts of your code or design decisions you've made.
4. If your PR has extensive changes, consider splitting it into smaller, related PRs. This reduces the cognitive load on the reviewers and speeds up the review process.

Please be responsive and participate in the discussion on your PR! This aids in clarifying any confusion or concerns and leads to quicker resolution and merging of your PR.

> Note: If your changes are not ready for merge but you want feedback, create a draft pull request.

Following these criteria will aid in quicker review and potential merging of your PR.
While the prerequisites above must be satisfied prior to having your pull request reviewed, the reviewer(s) may ask you to complete additional design work, tests, or other changes before your pull request can be ultimately accepted.

When you are ready to submit your changes, create a pull request:

> **Always** follow the [style guide](./STYLE.md) and [development workflow](./DEVELOPMENT_WORKFLOW.md) before submitting pull requests.

After you submit a pull request, it will be reviewed by the maintainers. They may ask you to make changes. Please respond to any comments and push your changes as a new commit.

> Note: Be sure to merge the latest from "upstream" before making a pull request:

```bash
git remote add upstream https://github.com/opentensor/bittensor.git
git fetch upstream
git merge upstream/<your-branch-name>
git push origin <your-branch-name>
```

#### Testing
Before making a PR for any code changes, please write adequate testing with unittest and/or pytest if it is warranted.  This is **mandatory** for new features and enhancements. See the [testing guide](./TESTING.md) for more complete information. 

You may also like to view the [/tests](https://github.com/opentensor/bittensor/tree/master/tests) for starter examples.

Here is a quick summary:
- **Running Tests**: Use `pytest` from the root directory of the Bittensor repository to run all tests. To run a specific test file or a specific test within a file, specify it directly (e.g., `pytest tests/test_wallet.py::test_create_new_coldkey`).
- **Writing Tests**: When writing tests, cover both the "happy path" and any potential error conditions. Use the `assert` statement to verify the expected behavior of a function.
- **Mocking**: Use the `unittest.mock` library to mock certain functions or objects when you need to isolate the functionality you're testing. This allows you to control the behavior of these functions or objects during testing.
- **Test Coverage**: Use the `pytest-cov` plugin to measure your test coverage. Aim for high coverage but also ensure your tests are meaningful and accurately represent the conditions under which your code will run.
- **Continuous Integration**: Bittensor uses GitHub Actions for continuous integration. Tests are automatically run every time you push changes to the repository. Check the "Actions" tab of the Bittensor GitHub repository to view the results.

Remember, testing is crucial for maintaining code health, catching issues early, and facilitating the addition of new features or refactoring of existing code.

#### Addressing Feedback

After submitting your pull request, expect comments and reviews from other contributors. You can add more commits to your pull request by committing them locally and pushing to your fork.

You are expected to reply to any review comments before your pull request is merged. You may update the code or reject the feedback if you do not agree with it, but you should express so in a reply. If there is outstanding feedback and you are not actively working on it, your pull request may be closed.

#### Squashing Commits

If your pull request contains fixup commits (commits that change the same line of code repeatedly) or too fine-grained commits, you may be asked to [squash](https://git-scm.com/docs/git-rebase#_interactive_mode) your commits before it will be reviewed. The basic squashing workflow is shown below.

    git checkout your_branch_name
    git rebase -i HEAD~n
    # n is normally the number of commits in the pull request.
    # Set commits (except the one in the first line) from 'pick' to 'squash', save and quit.
    # On the next screen, edit/refine commit messages.
    # Save and quit.
    git push -f # (force push to GitHub)

Please update the resulting commit message, if needed. It should read as a coherent message. In most cases, this means not just listing the interim commits.

If your change contains a merge commit, the above workflow may not work and you will need to remove the merge commit first. See the next section for details on how to rebase.

Please refrain from creating several pull requests for the same change. Use the pull request that is already open (or was created earlier) to amend changes. This preserves the discussion and review that happened earlier for the respective change set.

The length of time required for peer review is unpredictable and will vary from pull request to pull request.

#### Refactoring

Refactoring is a necessary part of any software project's evolution. The following guidelines cover refactoring pull requests for the Bittensor project.

There are three categories of refactoring: code-only moves, code style fixes, and code refactoring. In general, refactoring pull requests should not mix these three kinds of activities in order to make refactoring pull requests easy to review and uncontroversial. In all cases, refactoring PRs must not change the behaviour of code within the pull request (bugs must be preserved as is).

Project maintainers aim for a quick turnaround on refactoring pull requests, so where possible keep them short, uncomplex and easy to verify.

Pull requests that refactor the code should not be made by new contributors. It requires a certain level of experience to know where the code belongs to and to understand the full ramification (including rebase effort of open pull requests). Trivial pull requests or pull requests that refactor the code with no clear benefits may be immediately closed by the maintainers to reduce unnecessary workload on reviewing.

#### Peer Review

Anyone may participate in peer review which is expressed by comments in the pull request. Typically reviewers will review the code for obvious errors, as well as test out the patch set and opine on the technical merits of the patch. Project maintainers take into account the peer review when determining if there is consensus to merge a pull request (remember that discussions may have taken place elsewhere, not just on GitHub). The following language is used within pull-request comments:

- ACK means "I have tested the code and I agree it should be merged";
- NACK means "I disagree this should be merged", and must be accompanied by sound technical justification. NACKs without accompanying reasoning may be disregarded;
- utACK means "I have not tested the code, but I have reviewed it and it looks OK, I agree it can be merged";
- Concept ACK means "I agree in the general principle of this pull request";
- Nit refers to trivial, often non-blocking issues.

Reviewers should include the commit(s) they have reviewed in their comments. This can be done by copying the commit SHA1 hash.

A pull request that changes consensus-critical code is considerably more involved than a pull request that adds a feature to the wallet, for example. Such patches must be reviewed and thoroughly tested by several reviewers who are knowledgeable about the changed subsystems. Where new features are proposed, it is helpful for reviewers to try out the patch set on a test network and indicate that they have done so in their review. Project maintainers will take this into consideration when merging changes.

For a more detailed description of the review process, see the [Code Review Guidelines](CODE_REVIEW_DOCS.md).

### Reporting Bugs

This section guides you through submitting a bug report for Bittensor. Following these guidelines helps maintainers and the community understand your report :pencil:, reproduce the behavior :computer: :computer:, and find related reports :mag_right:.

When you are creating a bug report, please [include as many details as possible](#how-do-i-submit-a-good-bug-report).

> **Note:** If you find a **Closed** issue that seems like it is the same thing that you're experiencing, open a new issue and include a link to the original issue in the body of your new one.

#### Before Submitting A Bug Report

* **Check the [debugging guide](./DEBUGGING.md).** You might be able to find the cause of the problem and fix things yourself. Most importantly, check if you can reproduce the problem in the latest version of Bittensor by updating to the latest Master branch changes.
* **Check the [Discord Server](https://discord.gg/7wvFuPJZgq)** and ask in [#finney-issues](https://discord.com/channels/799672011265015819/1064247007688007800) or [#subnet-1-issues](https://discord.com/channels/799672011265015819/1096187495667998790).
* **Determine which repository the problem should be reported in**: if it has to do with your ML model, then it's likely [Bittensor](https://github.com/opentensor/bittensor). If you are having problems with your emissions or Blockchain, then it is in [subtensor](https://github.com/opentensor/subtensor) 

#### How Do I Submit A (Good) Bug Report?

Bugs are tracked as [GitHub issues](https://guides.github.com/features/issues/). You can find Bittensor's issues [here](https://github.com/opentensor/bittensor/issues). After you've determined which repository ([Bittensor](https://github.com/opentensor/bittensor) or [subtensor](https://github.com/opentensor/subtensor)) your bug is related to, create an issue on that repository.

Explain the problem and include additional details to help maintainers reproduce the problem:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible. For example, start by explaining how you started Bittensor, e.g. which command exactly you used in the terminal, or how you started Bittensor otherwise. When listing steps, **don't just say what you did, but explain how you did it**. For example, if you ran Bittensor with a set of custom configs, explain if you used a config file or command line arguments. 
* **Provide specific examples to demonstrate the steps**. Include links to files or GitHub projects, or copy/pasteable snippets, which you use in those examples. If you're providing snippets in the issue, use [Markdown code blocks](https://help.github.com/articles/markdown-basics/#multiple-lines).
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**
* **Include screenshots and animated GIFs** which show you following the described steps and clearly demonstrate the problem. You can use [this tool](https://www.cockos.com/licecap/) to record GIFs on macOS and Windows, and [this tool](https://github.com/colinkeenan/silentcast) or [this tool](https://github.com/GNOME/byzanz) on Linux.
* **If you're reporting that Bittensor crashed**, include a crash report with a stack trace from the operating system. On macOS, the crash report will be available in `Console.app` under "Diagnostic and usage information" > "User diagnostic reports". Include the crash report in the issue in a [code block](https://help.github.com/articles/markdown-basics/#multiple-lines), a [file attachment](https://help.github.com/articles/file-attachments-on-issues-and-pull-requests/), or put it in a [gist](https://gist.github.com/) and provide link to that gist.
* **If the problem is related to performance or memory**, include a CPU profile capture with your report, if you're using a GPU then include a GPU profile capture as well. Look into the [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) to look at memory usage of your model.
* **If the problem wasn't triggered by a specific action**, describe what you were doing before the problem happened and share more information using the guidelines below.

Provide more context by answering these questions:

* **Did the problem start happening recently** (e.g. after updating to a new version of Bittensor) or was this always a problem?
* If the problem started happening recently, **can you reproduce the problem in an older version of Bittensor?** 
* **Can you reliably reproduce the issue?** If not, provide details about how often the problem happens and under which conditions it normally happens.

Include details about your configuration and environment:

* **Which version of Bittensor are you using?** You can get the version by checking for `__version__` in [`bittensor/bittensor/__init.py`](https://github.com/opentensor/bittensor/blob/master/bittensor/__init__.py#L30). This is not sufficient. Also add the commit hash of the branch you are on.
* **What commit hash are you on?** You can get the exact commit hash by checking `git log` and pasting the full commit hash.
* **What's the name and version of the OS you're using**?
* **Are you running Bittensor in a virtual machine?** If so, which VM software are you using and which operating systems and versions are used for the host and the guest?
* **Are you running Bittensor in a dockerized container?** If so, have you made sure that your docker container contains your latest changes and is up to date with Master branch?
* **Are you using [local configuration files](https://opentensor.github.io/getting-started/configuration.html)** `config.yaml` to customize your Bittensor experiment? If so, provide the contents of that config file, preferably in a [code block](https://help.github.com/articles/markdown-basics/#multiple-lines) or with a link to a [gist](https://gist.github.com/).

### Suggesting Enhancements and Features

This section guides you through submitting an enhancement suggestion for Bittensor, including completely new features and minor improvements to existing functionality. Following these guidelines helps maintainers and the community understand your suggestion :pencil: and find related suggestions :mag_right:.

When you are creating an enhancement suggestion, please [include as many details as possible](#how-do-i-submit-a-good-enhancement-suggestion).

#### Before Submitting An Enhancement Suggestion

* **Check the [debugging guide](./DEBUGGING.md).** for tips — you might discover that the enhancement is already available. Most importantly, check if you're using the latest version of Bittensor by pulling the latest changes from the Master branch and if you can get the desired behavior by changing [Bittensor's config settings](https://opentensor.github.io/getting-started/configuration.html).
* **Determine which repository the problem should be reported in: if it has to do with your ML model, then it's likely [Bittensor](https://github.com/opentensor/bittensor). If you are having problems with your emissions or Blockchain, then it is in [subtensor](https://github.com/opentensor/subtensor) 

#### How To Submit A (Good) Feature Suggestion

Enhancement suggestions are tracked as [GitHub issues](https://guides.github.com/features/issues/). After you've determined which repository ([Bittensor](https://github.com/opentensor/bittensor) or [subtensor](https://github.com/opentensor/subtensor))  your enhancement suggestion is related to, create an issue on that repository and provide the following information:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Provide specific examples to demonstrate the steps**. Include copy/pasteable snippets which you use in those examples, as [Markdown code blocks](https://help.github.com/articles/markdown-basics/#multiple-lines).
* **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
* **Include screenshots and animated GIFs** which help you demonstrate the steps or point out the part of Bittensor which the suggestion is related to. You can use [this tool](https://www.cockos.com/licecap/) to record GIFs on macOS and Windows, and [this tool](https://github.com/colinkeenan/silentcast) or [this tool](https://github.com/GNOME/byzanz) on Linux.
* **Explain why this enhancement would be useful** to most Bittensor users.
* **List some other text editors or applications where this enhancement exists.**
* **Specify which version of Bittensor are you using?** You can get the exact version by checking for `__version__` in [`bittensor/bittensor/__init.py`](https://github.com/opentensor/bittensor/blob/master/bittensor/__init__.py#L30).
* **Specify the name and version of the OS you're using.**

Thank you for considering contributing to Bittensor! Any help is greatly appreciated along this journey to incentivize open and permissionless intelligence.
