name: Bug Fix Contribution
description: Use this template when contributing a bug fix.
labels: [bug, pull request]

body:
  - type: markdown
    attributes:
      value: |
        ### Requirements for Contributing a Bug Fix

        * Fill out the template below. Any pull request that does not include enough information to be reviewed in a timely manner may be closed at the maintainers' discretion.
        * The pull request must only fix an existing bug. To contribute other changes, you must use a different template. You can see all templates at <https://github.com/atom/.github/tree/master/.github/PULL_REQUEST_TEMPLATE>.
        * The pull request must update the test suite to demonstrate the changed functionality. For guidance, please see <https://flight-manual.atom.io/hacking-atom/sections/writing-specs/>.
        * After you create the pull request, all status checks must be pass before a maintainer reviews your contribution. For more details, please see <https://github.com/atom/.github/tree/master/CONTRIBUTING.md#pull-requests>.

  - type: input
    id: bug
    attributes:
      label: Identify the Bug
      description: Link to the issue describing the bug that you're fixing.
    validations:
      required: true

  - type: textarea
    id: change
    attributes:
      label: Description of the Change
      description: We must be able to understand the design of your change from this description.
    validations:
      required: true

  - type: textarea
    id: alternate
    attributes:
      label: Alternate Designs
      description: Explain what other alternates were considered and why the proposed version was selected.
    validations:
      required: false

  - type: textarea
    id: drawbacks
    attributes:
      label: Possible Drawbacks
      description: What are the possible side-effects or negative impacts of the code change?
    validations:
      required: false

  - type: textarea
    id: verification
    attributes:
      label: Verification Process
      description: What process did you follow to verify that the change has not introduced any regressions?
    validations:
      required: true

  - type: input
    id: release-notes
    attributes:
      label: Release Notes
      description: Please describe the changes in a single line that explains this improvement in terms that a user can understand.
    validations:
      required: true
