.. meta::
  :description: Composable Kernel documentation and API reference library
  :keywords: composable kernel, CK, ROCm, API, documentation

.. _contributing-to:

********************************************************************
Contributing to Composable Kernel
********************************************************************

Review the `Composable Kernel documentation <https://rocm.docs.amd.com/projects/composable_kernel/en/latest/>`_ before contributing to the Composable Kernel project. This documentation provides information about core concepts and configurations, as well as providing :doc:`steps for building Composable Kernel <install/Composable-Kernel-install>`. Some of this information is also available in the `Composable Kernel README <https://github.com/ROCm/composable_kernel/blob/develop/README.md>`_.

Consult the `AMD Developer Central portal <https://www.amd.com/en/developer.html>`_ for more information about AMD products.

Reporting issues
=================

Use `Github issues <https://github.com/ROCm/composable_kernel/issues>`_ to log and track issues and enhancement requests.

If you encounter an issue with the Composable Kernel library, search the existing GitHub issues to determine whether the problem has already been
reported. If it hasn't, submit a new issue that includes:

* A description of the problem, including what you observed, what you were expecting, and why this was an issue.
 
* Your configuration details, including the GPU, OS, and ROCm version, and any Docker image you used.

* The steps to reproduce the issue, including any CMake command you used to build the library, as well as the frequency of the issue.

* Any workarounds you've found and what you expect in a resolution. 


Contributing to the codebase
=============================

All external contributors to the Composable Kernel codebase must follow these guidelines:

* Use the correct branch: Use your own branch for your changes. Create your branch from the develop branch. 

* Describe your changes: Provide the motivation for the changes and a general description of all code changes.

* Add design documents for major changes: Major architectural changes must be accompanied by comprehensive design documents uploaded with your pull request. 

* Add inline documentation: Include relevant documentation and inline comments with your code changes.

* Link your pull request to related issues: Add links to any issues resolved by your changes in your pull request description.

* Verify and test the changes: Run all relevant existing tests and write new tests for any new functionality that isn't covered by existing tests.

* Provide performance numbers: Include documentation showing before and after performance numbers for any changes that potentially impact build times or run times. 

* Keep your branch up to date: Regularly rebase or merge the develop branch back into your feature branch. This should be done both prior to creating your pull request and during the review process.

* Ensure a manageable pull request size: Pull requests should be limited to approximately one thousand lines. If your changes significantly exceed one thousand lines, break them into smaller pull requests that can be reviewed independently.

* Use pre-commit hooks to adhere to the coding style: Composable Kernel's coding style is defined in `.clang-format <https://github.com/ROCm/composable_kernel/blob/develop/.clang-format>`_. Use the provided pre-commit hooks to run clang formatting and linting. Instructions on installing pre-commit hooks are available in the `README file <https://github.com/ROCm/composable_kernel/blob/develop/.clang-format>`_. 

Forks require an approver from AMD to trigger continuous integration (CI) testing. This approval process is necessary for security and resource management.

Depending on the complexity of your changes, an  AMD developer might need to pull your changes and perform additional fixes or modifications before merging. This collaborative approach ensures compatibility with internal systems and standards.

You can see a complete list of pull requests on the `Composable Kernel GitHub page <https://github.com/ROCm/composable_kernel/pulls>`_.

