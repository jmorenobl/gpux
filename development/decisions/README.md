# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records documenting important technical decisions made during the development of GPUX.

## Subdirectories

- `accepted/` - Accepted decisions that are in effect
- `proposed/` - Proposed decisions under review
- `rejected/` - Rejected decisions with rationale

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences. ADRs help maintain a record of why certain decisions were made and provide context for future developers.

## When to Create an ADR

Create an ADR when making decisions that:
- Affect the overall architecture or design
- Have significant technical or business impact
- Involve trade-offs between different approaches
- May be questioned or revisited in the future
- Set important precedents for future decisions

## ADR Lifecycle

1. **Proposal** - Create ADR in `proposed/` directory
2. **Review** - Team reviews and discusses the proposal
3. **Decision** - Accept, reject, or modify the proposal
4. **Implementation** - If accepted, implement the decision
5. **Archive** - Move to appropriate directory (`accepted/` or `rejected/`)

## ADR Template

Use the `TEMPLATE_DECISION.md` file as a starting point for new ADRs.

## ADR Numbering

Use sequential numbering: `ADR-001`, `ADR-002`, etc.

## ADR Status

- `[PROPOSED]` - Under review and discussion
- `[ACCEPTED]` - Decision has been accepted and implemented
- `[REJECTED]` - Decision has been rejected with rationale
- `[SUPERSEDED]` - Decision has been superseded by a newer ADR
- `[DEPRECATED]` - Decision is no longer relevant but kept for historical context

## ADR Categories

- **Architecture** - Overall system architecture and design
- **Technology** - Technology choices and tool selection
- **Integration** - Third-party integrations and APIs
- **Performance** - Performance-related decisions
- **Security** - Security and safety considerations
- **Usability** - User experience and interface decisions
- **Process** - Development process and workflow decisions

## ADR Review Process

1. **Initial Review** - Author creates ADR and requests review
2. **Team Discussion** - Team discusses pros, cons, and alternatives
3. **Decision** - Team reaches consensus or designated decision maker decides
4. **Documentation** - Update ADR with final decision and rationale
5. **Implementation** - Implement the decision and update related documentation

## Cross-References

- Link to related ideas in `../ideas/`
- Reference implementation tasks in `../implementation/`
- Connect to development phases in `../phases/`
- Link to research findings in `../research/`
- Reference GitHub issues and pull requests

## Best Practices

1. **Be Specific** - Clearly state what decision was made
2. **Provide Context** - Explain why the decision was necessary
3. **Document Alternatives** - Show what other options were considered
4. **Explain Consequences** - Describe the impact of the decision
5. **Keep Current** - Update ADRs when decisions change
6. **Make Discoverable** - Use clear titles and cross-references
