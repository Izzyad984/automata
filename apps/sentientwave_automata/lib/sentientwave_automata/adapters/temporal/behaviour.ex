defmodule SentientwaveAutomata.Adapters.Temporal.Behaviour do
  @moduledoc """
  Boundary for durable workflow orchestration.

  The runtime is backed only by the in-repo Temporal SDK adapter.
  """

  @callback start_workflow(workflow_name :: String.t(), input :: map(), opts :: keyword()) ::
              {:ok, %{workflow_id: String.t(), run_id: String.t(), status: atom()}}
              | {:error, term()}

  @callback signal_workflow(workflow_id :: String.t(), signal :: String.t(), payload :: map()) ::
              :ok | {:error, term()}

  @callback query_workflow(workflow_id :: String.t()) :: {:ok, map()} | {:error, term()}

  @callback start_agent_run(input :: map()) ::
              {:ok, %{workflow_id: String.t(), run_id: String.t(), status: atom()}}
              | {:error, term()}

  @callback signal_agent_run(workflow_id :: String.t(), payload :: map()) ::
              :ok | {:error, term()}

  @callback query_agent_run(workflow_id :: String.t()) :: {:ok, map()} | {:error, term()}
end
