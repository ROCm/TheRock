# TheRock RFCs

This directory is our temporary home for RFC documents which outline
project (ROCm) wide build, code organization, and engineering standards
processes that we are adopting. It is expected that it will eventually be
relocated to ROCm-org scope and encompass a more holistic process for managing
the evolution of the project.

Note that the project currently has an informal governance model that is
primarily driven by AMD with feedback opportunities from the community. As
such, at this phase, the documents here are mostly *advisory* for community
members and represent a public record of decisions that we have made which we
believe should be widely circulated as part of the project record. Please feel
free to open an issue or discussion thread if you feel that anything here would
benefit from further discussion.

## Index

- [RFC0001: BLAS Stack Build Improvements](./RFC0001-BLAS-Stack-Build-Improvements.md)
- [RFC0004: Test Orchestrator](./RFC0004-Test-Orchestrator.md)

## Adding an RFC

In order to add an RFC, create a new markdown file in this directory with a
name like `RFC0001-Some-Short-Title.md` and hyperlink it in the above index.

### Metadata

Include a
[YAML metadata block](https://github.blog/news-insights/product-news/viewing-yaml-metadata-in-your-documents/)
at the top of each RFC with the following fields:

- `author`: Full Name (GitHub handle)
- `created`: Creation date
- `modified`: Latest modification date
- `status`: One of "draft", "implemented", "obsolete"
