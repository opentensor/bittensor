# Contributing to Bittensor

The following is a set of guidelines for contributing to Bittensor, which are hosted in the [Opentensor Organization](https://github.com/opentensor) on GitHub. These are mostly guidelines, not rules. 
First, in terms of structure, there is no particular concept of "Bittensor Core developers" in the sense of privileged people. Open source often naturally revolves around a meritocracy where contributors earn trust from the developer community over time. Nevertheless, some hierarchy is necessary for practical purposes. As such, there are repository maintainers who are responsible for merging pull requests, the release cycle, and moderation.
## Table Of Contents

  1. [I don't want to read this whole thing, I just have a question!!!](#i-dont-want-to-read-this-whole-thing-i-just-have-a-question)
  1. [What should I know before I get started?](#what-should-i-know-before-i-get-started)
  1. [Getting Started](#getting-started)
  1. [Communication Channels](#communication-channels)
  1. [Contribution Workflow](#contribution-workflow)


## I don't want to read this whole thing I just have a question!!!

> **Note:** Please don't file an issue to ask a question. You'll get faster results by using the resources below.

We have an official Discord server where the community chimes in with helpful advice if you have questions. 
This is the fastest way to get an answer is the development team is active on Discord.

* [Official Bittensor Discord](https://discord.gg/7wvFuPJZgq)

## What should I know before I get started?
Bittensor is still in the Alpha stages, and as such you will likely run into some problems in deploying your model or installing Bittensor itself. If you run into an issue 
or resolve an issue yourself, feel free to create a pull request with a fix or with a fix to the documentation.

Additionally, note that the entire implementation of Bittensor cnosists of two separate repositories: [The core Bittensor code](https://github.com/opentensor/bittensor) and the Bittensor Blockchain [subtensor](https://github.com/opentensor/subtensor).

## Getting Started
New contributors are very welcome and needed.
Reviewing and testing is highly valued and the most effective way you can contribute as a new contributor. It also will teach you much more about the code and process than opening pull requests. 

Before you start contributing, familiarize yourself with the Bittensor Core build system and tests. Refer to the documentation in the repository on how to build Bittensor core and how to run the unit tests, functional tests, and fuzz tests.

There are many open issues of varying difficulty waiting to be fixed. If you're looking for somewhere to start contributing, check out the [good first issue](https://github.com/opentensor/bittensor/issues?page=3&q=is%3Aissue+is%3Aclosed) list or changes that are up for grabs. Some of them might no longer be applicable. So if you are interested, but unsure, you might want to leave a comment on the issue first.
### Good First Issue Label
The purpose of the good first issue label is to highlight which issues are suitable for a new contributor without a deep understanding of the codebase.

However, good first issues can be solved by anyone. If they remain unsolved for a longer time, a frequent contributor might address them.

You do not need to request permission to start working on an issue. However, you are encouraged to leave a comment if you are planning to work on it. This will help other contributors monitor which issues are actively being addressed and is also an effective way to request assistance if and when you need it.
### Beginner and Help-wanted Issues Label
You can start by looking through these `beginner` and `help-wanted` issues:

* Beginner issues - issues which should only require a few lines of code, and a test or two.
* Help wanted issues - issues which should be a bit more involved than `beginner` issues.

Both issue lists are sorted by total number of comments. While not perfect, number of comments is a reasonable proxy for impact a given change will have.

## Communication Channels
Most communication about Bittensor development happens on Discord channel.
Here's the link of Discord community.
https://discord.com/channels/799672011265015819/799672011814862902

And also here.
https://discord.com/channels/1120750674595024897/1120799375703162950

## Contribution Workflow

The codebase is maintained using the "contributor workflow" where everyone
without exception contributes patch proposals using "pull requests" (PRs). This
facilitates social contribution, easy testing and peer review.

To contribute a patch, the workflow is as follows:

  1. Fork repository ([only for the first time](https://github.com/opentensor/bittensor/fork))
  1. Create topic branch
  1. Commit patches

### Committing Patches

In general, [commits should be atomic](https://en.wikipedia.org/wiki/Atomic_commit#Atomic_commit_convention)
and diffs should be easy to read. For this reason, do not mix any formatting
fixes or code moves with actual code changes.

Make sure each individual commit is hygienic: that it builds successfully on its
own without warnings, errors, regressions, or test failures.

Commit messages should be verbose by default consisting of a short subject line
(50 chars max), a blank line and detailed explanatory text as separate
paragraph(s), unless the title alone is self-explanatory (like "Correct typo
in init.cpp") in which case a single title line is sufficient. Commit messages should be
helpful to people reading your code in the future, so explain the reasoning for
your decisions. Further explanation [here](https://chris.beams.io/posts/git-commit/).

If a particular commit references another issue, please add the reference. For
example: `refs #1234` or `fixes #4321`. Using the `fixes` or `closes` keywords
will cause the corresponding issue to be closed when the pull request is merged.

Commit messages should never contain any `@` mentions (usernames prefixed with "@").

Please refer to the [Git manual](https://git-scm.com/doc) for more information
about Git.

  - Push changes to your fork
  - Create pull request

### Reporting Bugs

This section guides you through submitting a bug report for Bittensor. Following these guidelines helps maintainers and the community understand your report :pencil:, reproduce the behavior :computer: :computer:, and find related reports :mag_right:.

Before creating bug reports, please check [this list](#before-submitting-a-bug-report) as you might find out that you don't need to create one. When you are creating a bug report, please [include as many details as possible](#how-do-i-submit-a-good-bug-report). Fill out [the required template](https://github.com/opentensor/bittensor/blob/master/.github/ISSUE_TEMPLATE/bug_report.md), the information it asks for helps us resolve issues faster.

> **Note:** If you find a **Closed** issue that seems like it is the same thing that you're experiencing, open a new issue and include a link to the original issue in the body of your new one.

#### Before Submitting A Bug Report

* **Check the [debugging guide](https://opentensor.github.io/bittensor/debugging.html).** You might be able to find the cause of the problem and fix things yourself. Most importantly, check if you can reproduce the problem in the latest version of Bittensor by updating to the latest Master branch changes.
* **Check the [Discord Server](https://discord.gg/7wvFuPJZgq)** and ask in #running-a-node channel or #contributions.
* **Determine which repository the problem should be reported in**: if it has to do with your ML model, then it's likely [Bittensor](https://github.com/opentensor/bittensor). If you are having problems with your emissions or Blockchain, then it is in [subtensor](https://github.com/opentensor/subtensor) 

#### How Do I Submit A (Good) Bug Report?

Bugs are tracked as [GitHub issues](https://guides.github.com/features/issues/). After you've determined which repository ([Bittensor](https://github.com/opentensor/bittensor) or [subtensor](https://github.com/opentensor/subtensor)) your bug is related to, create an issue on that repository and provide the following information by filling in [the template](https://github.com/opentensor/bittensor/blob/master/.github/ISSUE_TEMPLATE/bug_report.md).

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

* **Which version of Bittensor are you using?** You can get the exact version by checking for `__version__` in `[bittensor/bittensor/__init.py`](https://github.com/opentensor/bittensor/blob/master/bittensor/__init__.py#L9).
* **What's the name and version of the OS you're using**?
* **Are you running Bittensor in a virtual machine?** If so, which VM software are you using and which operating systems and versions are used for the host and the guest?
* **Are you running Bittensor in a dockerized container?** If so, have you made sure that your docker container contains your latest changes and is up to date with Master branch?
* **Are you using [local configuration files](https://opentensor.github.io/getting-started/configuration.html)** `config.yaml` to customize your Bittensor experiment? If so, provide the contents of that config file, preferably in a [code block](https://help.github.com/articles/markdown-basics/#multiple-lines) or with a link to a [gist](https://gist.github.com/).

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for Bittensor, including completely new features and minor improvements to existing functionality. Following these guidelines helps maintainers and the community understand your suggestion :pencil: and find related suggestions :mag_right:.

Before creating enhancement suggestions, please check [this list](#before-submitting-an-enhancement-suggestion) as you might find out that you don't need to create one. When you are creating an enhancement suggestion, please [include as many details as possible](#how-do-i-submit-a-good-enhancement-suggestion). Fill in [the template](https://github.com/opentensor/bittensor/blob/master/.github/ISSUE_TEMPLATE/feature_request.md), including the steps that you imagine you would take if the feature you're requesting existed.

#### Before Submitting An Enhancement Suggestion

* **Check the [debugging guide](https://opentensor.github.io/bittensor/debugging.html).** for tips — you might discover that the enhancement is already available. Most importantly, check if you're using the latest version of Bittensor by pulling the latest changes from the Master branch and if you can get the desired behavior by changing [Bittensor's config settings](https://opentensor.github.io/getting-started/configuration.html).
* **Determine which repository the problem should be reported in: if it has to do with your ML model, then it's likely [Bittensor](https://github.com/opentensor/bittensor). If you are having problems with your emissions or Blockchain, then it is in [subtensor](https://github.com/opentensor/subtensor) 

#### How Do I Submit A (Good) Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://guides.github.com/features/issues/). After you've determined which repository ([Bittensor](https://github.com/opentensor/bittensor) or [subtensor](https://github.com/opentensor/subtensor))  your enhancement suggestion is related to, create an issue on that repository and provide the following information:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Provide specific examples to demonstrate the steps**. Include copy/pasteable snippets which you use in those examples, as [Markdown code blocks](https://help.github.com/articles/markdown-basics/#multiple-lines).
* **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
* **Include screenshots and animated GIFs** which help you demonstrate the steps or point out the part of Bittensor which the suggestion is related to. You can use [this tool](https://www.cockos.com/licecap/) to record GIFs on macOS and Windows, and [this tool](https://github.com/colinkeenan/silentcast) or [this tool](https://github.com/GNOME/byzanz) on Linux.
* **Explain why this enhancement would be useful** to most Bittensor users.
* **List some other text editors or applications where this enhancement exists.**
* **Specify which version of Bittensor are you using?** You can get the exact version by checking for `__version__` in `[bittensor/bittensor/__init.py`](https://github.com/opentensor/bittensor/blob/master/bittensor/__init__.py#L18).
* **Specify the name and version of the OS you're using.**


### Creating the Pull Request

The title of the pull request should be prefixed by the component or area that
the pull request affects. 
Examples:
    scripts: Fix second script
    log: Fix typo in log message

The body of the pull request should contain sufficient description of *what* the
patch does, and even more importantly, *why*, with justification and reasoning.
You should include references to any discussions (for example, other issues or
mailing list discussions).

The description for a new pull request should not contain any `@` mentions. The
PR description will be included in the commit message when the PR is merged and
any users mentioned in the description will be annoyingly notified each time a
fork of BBittensor copies the merge. Instead, make any username mentions in a
subsequent comment to the PR.


### Work in Progress Changes and Requests for Comments

If a pull request is not to be considered for merging (yet), please
prefix the title with [WIP] or use [Tasks Lists](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#task-lists)
in the body of the pull request to indicate tasks are pending.

### Address Feedback

At this stage, one should expect comments and review from other contributors. You
can add more commits to your pull request by committing them locally and pushing
to your fork.

You are expected to reply to any review comments before your pull request is
merged. You may update the code or reject the feedback if you do not agree with
it, but you should express so in a reply. If there is outstanding feedback and
you are not actively working on it, your pull request may be closed.

Please refer to the [peer review](#peer-review) section below for more details.

### Squashing Commits

If your pull request contains fixup commits (commits that change the same line of code repeatedly) or too fine-grained
commits, you may be asked to [squash](https://git-scm.com/docs/git-rebase#_interactive_mode) your commits
before it will be reviewed. The basic squashing workflow is shown below.

    git checkout your_branch_name
    git rebase -i HEAD~n
    # n is normally the number of commits in the pull request.
    # Set commits (except the one in the first line) from 'pick' to 'squash', save and quit.
    # On the next screen, edit/refine commit messages.
    # Save and quit.
    git push -f # (force push to GitHub)

Please update the resulting commit message, if needed. It should read as a
coherent message. In most cases, this means not just listing the interim
commits.

If your change contains a merge commit, the above workflow may not work and you
will need to remove the merge commit first. See the next section for details on
how to rebase.

Please refrain from creating several pull requests for the same change.
Use the pull request that is already open (or was created earlier) to amend
changes. This preserves the discussion and review that happened earlier for
the respective change set.

The length of time required for peer review is unpredictable and will vary from
pull request to pull request.

### Rebasing Changes

When a pull request conflicts with the target branch, you may be asked to rebase it on top of the current target branch.

    git fetch https://github.com/opentensor/bittensor  # Fetch the latest upstream commit
    git rebase FETCH_HEAD  # Rebuild commits on top of the new base

This project aims to have a clean git history, where code changes are only made in non-merge commits. This simplifies
auditability because merge commits can be assumed to not contain arbitrary code changes. Merge commits should be signed,
and the resulting git tree hash must be deterministic and reproducible. The script in
[/contrib/verify-commits](/contrib/verify-commits) checks that.

After a rebase, reviewers are encouraged to sign off on the force push. This should be relatively straightforward with
the `git range-diff` tool explained in the [productivity
notes](/doc/productivity.md#diff-the-diffs-with-git-range-diff). To avoid needless review churn, maintainers will
generally merge pull requests that received the most review attention first.

Pull Request Philosophy
-----------------------

Patchsets should always be focused. For example, a pull request could add a
feature, fix a bug, or refactor code; but not a mixture. Please also avoid super
pull requests which attempt to do too much, are overly large, or overly complex
as this makes review difficult.


### Features

When adding a new feature, thought must be given to the long term technical debt
and maintenance that feature may require after inclusion. Before proposing a new
feature that will require maintenance, please consider if you are willing to
maintain it (including bug fixing). If features get orphaned with no maintainer
in the future, they may be removed by the Repository Maintainer.


### Refactoring

Refactoring is a necessary part of any software project's evolution. The
following guidelines cover refactoring pull requests for the project.

There are three categories of refactoring: code-only moves, code style fixes, and
code refactoring. In general, refactoring pull requests should not mix these
three kinds of activities in order to make refactoring pull requests easy to
review and uncontroversial. In all cases, refactoring PRs must not change the
behaviour of code within the pull request (bugs must be preserved as is).

Project maintainers aim for a quick turnaround on refactoring pull requests, so
where possible keep them short, uncomplex and easy to verify.

Pull requests that refactor the code should not be made by new contributors. It
requires a certain level of experience to know where the code belongs to and to
understand the full ramification (including rebase effort of open pull requests).

Trivial pull requests or pull requests that refactor the code with no clear
benefits may be immediately closed by the maintainers to reduce unnecessary
workload on reviewing.



### Peer Review

Anyone may participate in peer review which is expressed by comments in the pull
request. Typically reviewers will review the code for obvious errors, as well as
test out the patch set and opine on the technical merits of the patch. Project
maintainers take into account the peer review when determining if there is
consensus to merge a pull request (remember that discussions may have been
spread out over GitHub, mailing list and IRC discussions).

Code review is a burdensome but important part of the development process, and
as such, certain types of pull requests are rejected. In general, if the
**improvements** do not warrant the **review effort** required, the PR has a
high chance of being rejected. It is up to the PR author to convince the
reviewers that the changes warrant the review effort, and if reviewers are
"Concept NACK'ing" the PR, the author may need to present arguments and/or do
research backing their suggested changes.

#### Conceptual Review

A review can be a conceptual review, where the reviewer leaves a comment
 * `Concept (N)ACK`, meaning "I do (not) agree with the general goal of this pull
   request",
 * `Approach (N)ACK`, meaning `Concept ACK`, but "I do (not) agree with the
   approach of this change".

A `NACK` needs to include a rationale why the change is not worthwhile.
NACKs without accompanying reasoning may be disregarded.

#### Code Review

After conceptual agreement on the change, code review can be provided. A review
begins with `ACK BRANCH_COMMIT`, where `BRANCH_COMMIT` is the top of the PR
branch, followed by a description of how the reviewer did the review. The
following language is used within pull request comments:

  - "I have tested the code", involving change-specific manual testing in
    addition to running the unit, functional, or fuzz tests, and in case it is
    not obvious how the manual testing was done, it should be described;
  - "I have not tested the code, but I have reviewed it and it looks
    OK, I agree it can be merged";
  - A "nit" refers to a trivial, often non-blocking issue.

Project maintainers reserve the right to weigh the opinions of peer reviewers
using common sense judgement and may also weigh based on merit. Reviewers that
have demonstrated a deeper commitment and understanding of the project over time
or who have clear domain expertise may naturally have more weight, as one would
expect in all walks of life.

Where a patch set affects consensus-critical code, the bar will be much
higher in terms of discussion and peer review requirements, keeping in mind that
mistakes could be very costly to the wider community. This includes refactoring
of consensus-critical code.

Where a patch set proposes to change the Bittensor consensus, it must have been
discussed extensively on the mailing list and IRC, be accompanied by a widely
discussed BIP and have a generally widely perceived technical consensus of being
a worthwhile change based on the judgement of the maintainers.

### Finding Reviewers

As most reviewers are themselves developers with their own projects, the review
process can be quite lengthy, and some amount of patience is required. If you find
that you've been waiting for a pull request to be given attention for several
months, there may be a number of reasons for this, some of which you can do something
about:

  - It may be because of a feature freeze due to an upcoming release. During this time,
    only bug fixes are taken into consideration. If your pull request is a new feature,
    it will not be prioritized until after the release. Wait for the release.
  - It may be because the changes you are suggesting do not appeal to people. Rather than
    nits and critique, which require effort and means they care enough to spend time on your
    contribution, thundering silence is a good sign of widespread (mild) dislike of a given change
    (because people don't assume *others* won't actually like the proposal). Don't take
    that personally, though! Instead, take another critical look at what you are suggesting
    and see if it: changes too much, is too broad, doesn't adhere to the
    [developer notes](doc/developer-notes.md), is dangerous or insecure, is messily written, etc.
    Identify and address any of the issues you find. Then ask e.g. on IRC if someone could give
    their opinion on the concept itself.
  - It may be because your code is too complex for all but a few people, and those people
    may not have realized your pull request even exists. A great way to find people who
    are qualified and care about the code you are touching is the
    [Git Blame feature](https://docs.github.com/en/github/managing-files-in-a-repository/managing-files-on-github/tracking-changes-in-a-file). Simply
    look up who last modified the code you are changing and see if you can find
    them and give them a nudge. Don't be incessant about the nudging, though.
  - Finally, if all else fails, ask on IRC or elsewhere for someone to give your pull request
    a look. If you think you've been waiting for an unreasonably long time (say,
    more than a month) for no particular reason (a few lines changed, etc.),
    this is totally fine. Try to return the favor when someone else is asking
    for feedback on their code, and the universe balances out.
  - Remember that the best thing you can do while waiting is give review to others!

