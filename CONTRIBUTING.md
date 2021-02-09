# Contributing to Bittensor

The following is a set of guidelines for contributing to Bittensor, which are hosted in the [Opentensor Organization](https://github.com/opentensor) on GitHub. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

#### Table Of Contents

[Code of Conduct](#code-of-conduct)

[I don't want to read this whole thing, I just have a question!!!](#i-dont-want-to-read-this-whole-thing-i-just-have-a-question)

[What should I know before I get started?](#what-should-i-know-before-i-get-started)

[How Can I Contribute?](#how-can-i-contribute)
  * [Reporting Bugs](#reporting-bugs)
  * [Suggesting Enhancements](#suggesting-enhancements)
  * [Your First Code Contribution](#your-first-code-contribution)
  * [Pull Requests](#pull-requests)

## I don't want to read this whole thing I just have a question!!!

> **Note:** Please don't file an issue to ask a question. You'll get faster results by using the resources below.

We have an official Discord server where the community chimes in with helpful advice if you have questions. 
This is the fastest way to get an answer is the development team is active on Discord.

* [Official Bittensor Discord](https://discord.gg/7wvFuPJZgq)

## What should I know before I get started?
Bittensor is still in the Alpha 2.0 stages, and as such you will likely run into some problems in deploying your model or installing Bittensor itself. If you run into an issue 
or resolve an issue yourself, feel free to create a pull request with a fix or with a fix to the documentation. The documentation repository can be found [here](https://github.com/opentensor/opentensor.github.io). 

Additionally, note that the entire implementation of Bittensor cnosists of two separate repositories: [The core Bittensor code](https://github.com/opentensor/bittensor) and the Bittensor Blockchain [subtensor[(https://github.com/opentensor/subtensor).

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for Bittensor. Following these guidelines helps maintainers and the community understand your report :pencil:, reproduce the behavior :computer: :computer:, and find related reports :mag_right:.

Before creating bug reports, please check [this list](#before-submitting-a-bug-report) as you might find out that you don't need to create one. When you are creating a bug report, please [include as many details as possible](#how-do-i-submit-a-good-bug-report). Fill out [the required template](https://github.com/opentensor/bittensor/.github/blob/master/.github/ISSUE_TEMPLATE/bug_report.md), the information it asks for helps us resolve issues faster.

> **Note:** If you find a **Closed** issue that seems like it is the same thing that you're experiencing, open a new issue and include a link to the original issue in the body of your new one.

#### Before Submitting A Bug Report

* **Check the [debugging guide](https://opentensor.github.io/bittensor/debugging.html).** You might be able to find the cause of the problem and fix things yourself. Most importantly, check if you can reproduce the problem in the latest version of Bittensor by updating to the latest Master branch changes.
* **Check the [Discord Server](https://discord.gg/7wvFuPJZgq) and ask in #running-a-node channel or #contributions.
* **Determine which repository the problem should be reported in: if it has to do with your ML model, then it's likely [Bittensor](https://github.com/opentensor/bittensor). If you are having problems with your emissions or Blockchain, then it is in [subtensor](https://github.com/opentensor/subtensor) 

#### How Do I Submit A (Good) Bug Report?

Bugs are tracked as [GitHub issues](https://guides.github.com/features/issues/). After you've determined which repository ([Bittensor](https://github.com/opentensor/bittensor) or [subtensor](https://github.com/opentensor/subtensor) ) your bug is related to, create an issue on that repository and provide the following information by filling in [the template](https://github.com/opentensor/bittensor/.github/blob/master/.github/ISSUE_TEMPLATE/bug_report.md).

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

* **Which version of Bittensor are you using?** You can get the exact version by checking for `__version__` in `[bittensor/bittensor/__init.py`](https://github.com/opentensor/bittensor/blob/master/bittensor/__init__.py#L18).
* **What's the name and version of the OS you're using**?
* **Are you running Bittensor in a virtual machine?** If so, which VM software are you using and which operating systems and versions are used for the host and the guest?
* **Are you running Bittensor in a dockerized container?** If so, have you made sure that your docker container contains your latest changes and is up to date with Master branch?
* **Are you using [local configuration files](https://opentensor.github.io/getting-started/configuration.html)** `config.yaml` to customize your Bittensor experiment? If so, provide the contents of that config file, preferably in a [code block](https://help.github.com/articles/markdown-basics/#multiple-lines) or with a link to a [gist](https://gist.github.com/).

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for Bittensor, including completely new features and minor improvements to existing functionality. Following these guidelines helps maintainers and the community understand your suggestion :pencil: and find related suggestions :mag_right:.

Before creating enhancement suggestions, please check [this list](#before-submitting-an-enhancement-suggestion) as you might find out that you don't need to create one. When you are creating an enhancement suggestion, please [include as many details as possible](#how-do-i-submit-a-good-enhancement-suggestion). Fill in [the template](https://github.com/opentensor/bittensor/.github/blob/master/.github/ISSUE_TEMPLATE/feature_request.md), including the steps that you imagine you would take if the feature you're requesting existed.

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

### Your First Code Contribution

Unsure where to begin contributing to Bittensor? You can start by looking through these `beginner` and `help-wanted` issues:

* [Beginner issues][beginner] - issues which should only require a few lines of code, and a test or two.
* [Help wanted issues][help-wanted] - issues which should be a bit more involved than `beginner` issues.

Both issue lists are sorted by total number of comments. While not perfect, number of comments is a reasonable proxy for impact a given change will have.

If you want to read about using Bittensor or developing models in Bittensor, the [Bittensor Documentation](https://opentensor.github.io) is free and available online. You can find the source to the manual in [opentensor/opentensor.github.io](http://github.com/opentensor/opentensor.github.io).

### Pull Requests

The process described here has several goals:

- Maintain Bittensor's quality
- Fix problems that are important to users
- Engage the community in working toward the best possible iteration of Bittensor
- Enable a sustainable system for Bittensor's maintainers to review contributions

Please follow these steps to have your contribution considered by the maintainers:

1. Follow all instructions in [the template](PULL_REQUEST_TEMPLATE.md)
2. Follow the [styleguides](#styleguides)
3. After you submit your pull request, verify that all [status checks](https://help.github.com/articles/about-status-checks/) are passing <details><summary>What if the status checks are failing?</summary>If a status check is failing, and you believe that the failure is unrelated to your change, please leave a comment on the pull request explaining why you believe the failure is unrelated. A maintainer will re-run the status check for you. If we conclude that the failure was a false positive, then we will open an issue to track that problem with our status check suite.</details>

While the prerequisites above must be satisfied prior to having your pull request reviewed, the reviewer(s) may ask you to complete additional design work, tests, or other changes before your pull request can be ultimately accepted.


