# Style Guide

A project’s long-term success rests (among other things) on its maintainability, and a maintainer has few tools more powerful than his or her project’s log. It’s worth taking the time to learn how to care for one properly. What may be a hassle at first soon becomes habit, and eventually a source of pride and productivity for all involved.

Most programming languages have well-established conventions as to what constitutes idiomatic style, i.e. naming, formatting and so on. There are variations on these conventions, of course, but most developers agree that picking one and sticking to it is far better than the chaos that ensues when everybody does their own thing.

# Table of Contents
1. [Code Style](#code-style)
2. [Naming Conventions](#naming-conventions)
3. [Git Commit Style](#git-commit-style)
4. [The Six Rules of a Great Commit](#the-six-rules-of-a-great-commit)
   - [1. Atomic Commits](#1-atomic-commits)
   - [2. Separate Subject from Body with a Blank Line](#2-separate-subject-from-body-with-a-blank-line)
   - [3. Limit the Subject Line to 50 Characters](#3-limit-the-subject-line-to-50-characters)
   - [4. Use the Imperative Mood in the Subject Line](#4-use-the-imperative-mood-in-the-subject-line)
   - [5. Wrap the Body at 72 Characters](#5-wrap-the-body-at-72-characters)
   - [6. Use the Body to Explain What and Why vs. How](#6-use-the-body-to-explain-what-and-why-vs-how)
5. [Tools Worth Mentioning](#tools-worth-mentioning)
   - [Using `--fixup`](#using---fixup)
   - [Interactive Rebase](#interactive-rebase)
6. [Pull Request and Squashing Commits Caveats](#pull-request-and-squashing-commits-caveats)


### Code style

#### General Style
Python's official style guide is PEP 8, which provides conventions for writing code for the main Python distribution. Here are some key points:

- `Indentation:` Use 4 spaces per indentation level.

- `Line Length:` Limit all lines to a maximum of 79 characters.

- `Blank Lines:` Surround top-level function and class definitions with two blank lines. Method definitions inside a class are surrounded by a single blank line.

- `Imports:` Imports should usually be on separate lines and should be grouped in the following order:

    - Standard library imports.
    - Related third party imports.
    - Local application/library specific imports.
- `Whitespace:` Avoid extraneous whitespace in the following situations:

    - Immediately inside parentheses, brackets or braces.
    - Immediately before a comma, semicolon, or colon.
    - Immediately before the open parenthesis that starts the argument list of a function call.
- `Comments:` Comments should be complete sentences and should be used to clarify code and are not a substitute for poorly written code.

#### For Python

- `List Comprehensions:` Use list comprehensions for concise and readable creation of lists.

- `Generators:` Use generators when dealing with large amounts of data to save memory.

- `Context Managers:` Use context managers (with statement) for resource management.

- `String Formatting:` Use f-strings for formatting strings in Python 3.6 and above.

- `Error Handling:` Use exceptions for error handling whenever possible.

#### More details

Use `black` to format your python code before commiting for consistency across such a large pool of contributors. Black's code [style](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html#code-style) ensures consistent and opinionated code formatting. It automatically formats your Python code according to the Black style guide, enhancing code readability and maintainability.

Key Features of Black:

    Consistency: Black enforces a single, consistent coding style across your project, eliminating style debates and allowing developers to focus on code logic.

    Readability: By applying a standard formatting style, Black improves code readability, making it easier to understand and collaborate on projects.

    Automation: Black automates the code formatting process, saving time and effort. It eliminates the need for manual formatting and reduces the likelihood of inconsistencies.

### Naming Conventions

- `Classes:` Class names should normally use the CapWords Convention.
- `Functions and Variables:` Function names should be lowercase, with words separated by underscores as necessary to improve readability. Variable names follow the same convention as function names.

- `Constants:` Constants are usually defined on a module level and written in all capital letters with underscores separating words.

- `Non-public Methods and Instance Variables:` Use a single leading underscore (_). This is a weak "internal use" indicator.

- `Strongly "private" methods and variables:` Use a double leading underscore (__). This triggers name mangling in Python.


### Git commit style

Here’s a model Git commit message when contributing:
```
Summarize changes in around 50 characters or less

More detailed explanatory text, if necessary. Wrap it to about 72
characters or so. In some contexts, the first line is treated as the
subject of the commit and the rest of the text as the body. The
blank line separating the summary from the body is critical (unless
you omit the body entirely); various tools like `log`, `shortlog`
and `rebase` can get confused if you run the two together.

Explain the problem that this commit is solving. Focus on why you
are making this change as opposed to how (the code explains that).
Are there side effects or other unintuitive consequences of this
change? Here's the place to explain them.

Further paragraphs come after blank lines.

 - Bullet points are okay, too

 - Typically a hyphen or asterisk is used for the bullet, preceded
   by a single space, with blank lines in between, but conventions
   vary here

If you use an issue tracker, put references to them at the bottom,
like this:

Resolves: #123
See also: #456, #789
```


## The six rules of a great commit.

#### 1. Atomic Commits
An “atomic” change revolves around one task or one fix.

Atomic Approach
 - Commit each fix or task as a separate change
 - Only commit when a block of work is complete
 - Commit each layout change separately
 - Joint commit for layout file, code behind file, and additional resources

Benefits

- Easy to roll back without affecting other changes
- Easy to make other changes on the fly
- Easy to merge features to other branches

#### Avoid trivial commit messages

Commit messages like "fix", "fix2", or "fix3" don't provide any context or clear understanding of what changes the commit introduces. Here are some examples of good vs. bad commit messages:

**Bad Commit Message:** 

    $ git commit -m "fix"

**Good Commit Message:**

    $ git commit -m "Fix typo in README file"

> **Caveat**: When working with new features, an atomic commit will often consist of multiple files, since a layout file, code behind file, and additional resources may have been added/modified. You don’t want to commit all of these separately, because if you had to roll back the application to a state before the feature was added, it would involve multiple commit entries, and that can get confusing

#### 2. Separate subject from body with a blank line

Not every commit requires both a subject and a body. Sometimes a single line is fine, especially when the change is so simple that no further context is necessary. 

For example:

    Fix typo in introduction to user guide

Nothing more need be said; if the reader wonders what the typo was, she can simply take a look at the change itself, i.e. use     git show or git diff or git log -p.

If you’re committing something like this at the command line, it’s easy to use the -m option to git commit:

    $ git commit -m"Fix typo in introduction to user guide"

However, when a commit merits a bit of explanation and context, you need to write a body. For example:

    Derezz the master control program

    MCP turned out to be evil and had become intent on world domination.
    This commit throws Tron's disc into MCP (causing its deresolution)
    and turns it back into a chess game.

Commit messages with bodies are not so easy to write with the -m option. You’re better off writing the message in a proper text editor. [See Pro Git](https://git-scm.com/book/en/v2/Customizing-Git-Git-Configuration).

In any case, the separation of subject from body pays off when browsing the log. Here’s the full log entry:

    $ git log
    commit 42e769bdf4894310333942ffc5a15151222a87be
    Author: Kevin Flynn <kevin@flynnsarcade.com>
    Date:   Fri Jan 01 00:00:00 1982 -0200
    
     Derezz the master control program
    
     MCP turned out to be evil and had become intent on world domination.
     This commit throws Tron's disc into MCP (causing its deresolution)
     and turns it back into a chess game.


#### 3. Limit the subject line to 50 characters
50 characters is not a hard limit, just a rule of thumb. Keeping subject lines at this length ensures that they are readable, and forces the author to think for a moment about the most concise way to explain what’s going on.

GitHub’s UI is fully aware of these conventions. It will warn you if you go past the 50 character limit. Git will truncate any subject line longer than 72 characters with an ellipsis, thus keeping it to 50 is best practice.

#### 4. Use the imperative mood in the subject line
Imperative mood just means “spoken or written as if giving a command or instruction”. A few examples:

    Clean your room
    Close the door
    Take out the trash

Each of the seven rules you’re reading about right now are written in the imperative (“Wrap the body at 72 characters”, etc.).

The imperative can sound a little rude; that’s why we don’t often use it. But it’s perfect for Git commit subject lines. One reason for this is that Git itself uses the imperative whenever it creates a commit on your behalf.

For example, the default message created when using git merge reads:

    Merge branch 'myfeature'

And when using git revert:

    Revert "Add the thing with the stuff"

    This reverts commit cc87791524aedd593cff5a74532befe7ab69ce9d.

Or when clicking the “Merge” button on a GitHub pull request:

    Merge pull request #123 from someuser/somebranch

So when you write your commit messages in the imperative, you’re following Git’s own built-in conventions. For example:

    Refactor subsystem X for readability
    Update getting started documentation
    Remove deprecated methods
    Release version 1.0.0

Writing this way can be a little awkward at first. We’re more used to speaking in the indicative mood, which is all about reporting facts. That’s why commit messages often end up reading like this:

    Fixed bug with Y
    Changing behavior of X

And sometimes commit messages get written as a description of their contents:

    More fixes for broken stuff
    Sweet new API methods

To remove any confusion, here’s a simple rule to get it right every time.

**A properly formed Git commit subject line should always be able to complete the following sentence:**

    If applied, this commit will <your subject line here>

For example:

    If applied, this commit will refactor subsystem X for readability
    If applied, this commit will update getting started documentation
    If applied, this commit will remove deprecated methods
    If applied, this commit will release version 1.0.0
    If applied, this commit will merge pull request #123 from user/branch

#### 5. Wrap the body at 72 characters
Git never wraps text automatically. When you write the body of a commit message, you must mind its right margin, and wrap text manually.

The recommendation is to do this at 72 characters, so that Git has plenty of room to indent text while still keeping everything under 80 characters overall.

A good text editor can help here. It’s easy to configure Vim, for example, to wrap text at 72 characters when you’re writing a Git commit.

#### 6. Use the body to explain what and why vs. how
This [commit](https://github.com/bitcoin/bitcoin/commit/eb0b56b19017ab5c16c745e6da39c53126924ed6) from Bitcoin Core is a great example of explaining what changed and why:

```
commit eb0b56b19017ab5c16c745e6da39c53126924ed6
Author: Pieter Wuille <pieter.wuille@gmail.com>
Date:   Fri Aug 1 22:57:55 2014 +0200

   Simplify serialize.h's exception handling

   Remove the 'state' and 'exceptmask' from serialize.h's stream
   implementations, as well as related methods.

   As exceptmask always included 'failbit', and setstate was always
   called with bits = failbit, all it did was immediately raise an
   exception. Get rid of those variables, and replace the setstate
   with direct exception throwing (which also removes some dead
   code).

   As a result, good() is never reached after a failure (there are
   only 2 calls, one of which is in tests), and can just be replaced
   by !eof().

   fail(), clear(n) and exceptions() are just never called. Delete
   them.
```

Take a look at the [full diff](https://github.com/bitcoin/bitcoin/commit/eb0b56b19017ab5c16c745e6da39c53126924ed6) and just think how much time the author is saving fellow and future committers by taking the time to provide this context here and now. If he didn’t, it would probably be lost forever.

In most cases, you can leave out details about how a change has been made. Code is generally self-explanatory in this regard (and if the code is so complex that it needs to be explained in prose, that’s what source comments are for). Just focus on making clear the reasons why you made the change in the first place—the way things worked before the change (and what was wrong with that), the way they work now, and why you decided to solve it the way you did.

The future maintainer that thanks you may be yourself!



#### Tools worth mentioning

##### Using `--fixup`

If you've made a commit and then realize you've missed something or made a minor mistake, you can use the `--fixup` option. 

For example, suppose you've made a commit with a hash `9fceb02`. Later, you realize you've left a debug statement in your code. Instead of making a new commit titled "remove debug statement" or "fix", you can do the following:

    $ git commit --fixup 9fceb02

This will create a new commit to fix the issue, with a message like "fixup! The original commit message".

##### Interactive Rebase

Interactive rebase, or `rebase -i`, can be used to squash these fixup commits into the original commits they're fixing, which cleans up your commit history. You can use the `autosquash` option to automatically squash any commits marked as "fixup" into their target commits.

For example:

    $ git rebase -i --autosquash HEAD~5

This command starts an interactive rebase for the last 5 commits (`HEAD~5`). Any commits marked as "fixup" will be automatically moved to squash with their target commits.

The benefit of using `--fixup` and interactive rebase is that it keeps your commit history clean and readable. It groups fixes with the commits they are related to, rather than having a separate "fix" commit that might not make sense to other developers (or even to you) in the future.


---

#### Pull Request and Squashing Commits Caveats

While atomic commits are great for development and for understanding the changes within the branch, the commit history can get messy when merging to the main branch. To keep a cleaner and more understandable commit history in our main branch, we encourage squashing all the commits of a PR into one when merging.

This single commit should provide an overview of the changes that the PR introduced. It should follow the guidelines for atomic commits (an atomic commit is complete, self-contained, and understandable) but on the scale of the entire feature, task, or fix that the PR addresses. This approach combines the benefits of atomic commits during development with a clean commit history in our main branch.

Here is how you can squash commits:

```bash
git rebase -i HEAD~n
```

where `n` is the number of commits to squash. After running the command, replace `pick` with `squash` for the commits you want to squash into the previous commit. This will combine the commits and allow you to write a new commit message.

In this context, an atomic commit message could look like:

```
Add feature X

This commit introduces feature X which does A, B, and C. It adds 
new files for layout, updates the code behind the file, and introduces
new resources. This change is important because it allows users to 
perform task Y more efficiently. 

It includes:
- Creation of new layout file
- Updates in the code-behind file
- Addition of new resources

Resolves: #123
```

In your PRs, remember to detail what the PR is introducing or fixing. This will be helpful for reviewers to understand the context and the reason behind the changes. 
