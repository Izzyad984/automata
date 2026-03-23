# Architecture

## Platform Components
- `sentientwave_automata_web`: Phoenix API/UI and ingress surface
- `sentientwave_automata`: orchestration domain, edition gates, adapters, seat manager
- Matrix adapter boundary for inbound/outbound room events
- Temporal adapter boundary for durable workflow execution

## Execution Flow
1. Client posts workflow request to `/api/v1/workflows`.
2. Orchestrator validates payload and entitlement.
3. Temporal adapter starts a durable workflow run.
4. Matrix adapter posts status updates to room(s).
5. Workflow summaries are stored and queryable via API.

## Elixir-First Boundaries
- `SentientwaveAutomata.Orchestrator`
- `SentientwaveAutomata.Policy.Entitlements`
- `SentientwaveAutomata.Adapters.Matrix.*`
- `SentientwaveAutomata.Adapters.Temporal.*`
- `SentientwaveAutomata.Agents.*`

## Temporal Integration
- In-repo Elixir Temporal workers own agent runs, governance proposals, scheduled tasks, and generic conversation workflows
- Phoenix and Matrix stay on the control plane: they persist requests, then start or signal Temporal workflows
- Workflow summaries are stored durably in Postgres for API and admin UI introspection
