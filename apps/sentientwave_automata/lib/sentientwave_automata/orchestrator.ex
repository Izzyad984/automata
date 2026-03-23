defmodule SentientwaveAutomata.Orchestrator do
  @moduledoc """
  Control-plane boundary for generic conversation workflows.
  """

  import Ecto.Query, warn: false

  alias SentientwaveAutomata.Orchestration.Workflow
  alias SentientwaveAutomata.Policy.Entitlements
  alias SentientwaveAutomata.Repo
  alias SentientwaveAutomata.Temporal

  @spec start_workflow(map()) :: {:ok, Workflow.t()} | {:error, term()}
  def start_workflow(
        %{room_id: room_id, objective: objective, requested_by: requested_by} = attrs
      ) do
    with :ok <- validate_payload(attrs),
         true <-
           Entitlements.allowed?(:basic_orchestration, attrs) || {:error, :feature_not_enabled},
         {:ok, workflow} <- create_workflow(attrs),
         {:ok, temporal} <-
           temporal_adapter().start_workflow(
             "conversation_workflow",
             %{
               workflow_id: workflow.workflow_id,
               workflow_summary_id: workflow.id,
               attrs: attrs
             },
             workflow_id: workflow.workflow_id
           ),
         {:ok, updated_workflow} <-
           update_workflow(workflow, %{
             run_id: temporal.run_id,
             status: temporal.status,
             metadata:
               Map.merge(workflow.metadata || %{}, %{
                 "temporal_source" => "temporal_sdk",
                 "room_id" => room_id,
                 "objective" => objective,
                 "requested_by" => requested_by
               })
           }) do
      {:ok, updated_workflow}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  @spec list_workflows() :: [Workflow.t()]
  def list_workflows do
    Workflow
    |> order_by([workflow], desc: workflow.inserted_at, desc: workflow.workflow_id)
    |> Repo.all()
  end

  @spec get_workflow_by_workflow_id(binary()) :: Workflow.t() | nil
  def get_workflow_by_workflow_id(workflow_id) when is_binary(workflow_id) do
    Repo.get_by(Workflow, workflow_id: workflow_id)
  end

  @spec update_workflow(Workflow.t(), map()) :: {:ok, Workflow.t()} | {:error, term()}
  def update_workflow(%Workflow{} = workflow, attrs) when is_map(attrs) do
    workflow
    |> Workflow.changeset(attrs)
    |> Repo.update()
  end

  defp create_workflow(attrs) do
    workflow_id = Temporal.generated_workflow_id("conversation_workflow")

    %Workflow{}
    |> Workflow.changeset(%{
      workflow_id: workflow_id,
      status: :running,
      room_id: attrs.room_id,
      objective: attrs.objective,
      requested_by: attrs.requested_by,
      metadata: %{
        "edition" => to_string(Map.get(attrs, :edition, :community))
      }
    })
    |> Repo.insert()
  end

  defp validate_payload(%{room_id: room_id, objective: objective, requested_by: requested_by}) do
    if Enum.all?([room_id, objective, requested_by], &(is_binary(&1) and String.trim(&1) != "")) do
      :ok
    else
      {:error, :invalid_payload}
    end
  end

  defp temporal_adapter do
    Application.get_env(
      :sentientwave_automata,
      :temporal_adapter,
      SentientwaveAutomata.Adapters.Temporal.Runtime
    )
  end
end
