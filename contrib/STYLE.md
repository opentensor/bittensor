# Style Guide

A project’s long-term success rests (among other things) on its maintainability, and a maintainer has few tools more powerful than his or her project’s log. It’s worth taking the time to learn how to care for one properly. What may be a hassle at first soon becomes habit, and eventually a source of pride and productivity for all involved.

Most programming languages have well-established conventions as to what constitutes idiomatic style, i.e. naming, formatting and so on. There are variations on these conventions, of course, but most developers agree that picking one and sticking to it is far better than the chaos that ensues when everybody does their own thing.

### Code style
Use `black` to format your python code before commiting for consistency across such a large pool of contributors. Black's code [style](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html#code-style) ensures consistent and opinionated code formatting. It automatically formats your Python code according to the Black style guide, enhancing code readability and maintainability.

Key Features of Black:

    Consistency: Black enforces a single, consistent coding style across your project, eliminating style debates and allowing developers to focus on code logic.

    Readability: By applying a standard formatting style, Black improves code readability, making it easier to understand and collaborate on projects.

    Automation: Black automates the code formatting process, saving time and effort. It eliminates the need for manual formatting and reduces the likelihood of inconsistencies.

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

*Caveat*: When working with new features, an atomic commit will often consist of multiple files, since a layout file, code behind file, and additional resources may have been added/modified. You don’t want to commit all of these separately, because if you had to roll back the application to a state before the feature was added, it would involve multiple commit entries, and that can get confusing

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