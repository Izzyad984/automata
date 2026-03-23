defmodule SentientwaveAutomataTemporal.Bootstrap do
  @moduledoc """
  Verifies Temporal availability and reconciles Temporal-owned workflow state.
  """

  use GenServer

  alias SentientwaveAutomata.Temporal
  require Logger

  @reconcile_interval_ms 30_000

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    verify_temporal!()
    send(self(), :reconcile)
    {:ok, %{}}
  end

  @impl true
  def handle_info(:reconcile, state) do
    reconcile()
    Process.send_after(self(), :reconcile, @reconcile_interval_ms)
    {:noreply, state}
  end

  defp verify_temporal! do
    case TemporalSdk.Cluster.is_started(Temporal.cluster()) do
      true -> :ok
      {:error, reason} -> raise "Temporal cluster is not started: #{inspect(reason)}"
      other -> raise "Temporal cluster is not ready: #{inspect(other)}"
    end

    health_workflow_id =
      Temporal.generated_workflow_id("temporal_healthcheck_#{node()}")

    case TemporalSdk.start_workflow(
           Temporal.cluster(),
           Temporal.workflow_task_queue(),
           SentientwaveAutomataTemporal.HealthWorkflow,
           namespace: Temporal.namespace(),
           workflow_id: health_workflow_id,
           wait: 15_000,
           input: [%{"health" => true}]
         ) do
      {response, _result} when is_map(response) ->
        :ok

      {:ok, _response, _awaited} ->
        :ok

      {:error, reason} ->
        raise "Temporal health workflow failed to start: #{inspect(reason)}"

      other ->
        raise "Temporal health workflow returned unexpected result: #{inspect(other)}"
    end
  end

  defp reconcile do
    run_if_exported(SentientwaveAutomata.Agents, :mark_orphaned_runs_failed, [])
    run_if_exported(SentientwaveAutomata.Agents.ScheduledTaskReconciler, :reconcile, [])
    run_if_exported(SentientwaveAutomata.Governance.Workflow, :reconcile_open_proposals, [])
  rescue
    error ->
      Logger.warning("temporal_bootstrap reconcile_failed error=#{Exception.message(error)}")

      :ok
  end

  defp run_if_exported(module, function, args) do
    if function_exported?(module, function, length(args)) do
      apply(module, function, args)
    else
      :ok
    end
  end
end
