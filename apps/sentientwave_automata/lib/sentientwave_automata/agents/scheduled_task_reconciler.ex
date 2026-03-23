defmodule SentientwaveAutomata.Agents.ScheduledTaskReconciler do
  @moduledoc """
  Control-plane reconciler that ensures persisted scheduled tasks are mirrored
  into Temporal-managed scheduler workflows.
  """

  use GenServer

  alias SentientwaveAutomata.Agents
  alias SentientwaveAutomata.Agents.ScheduledTask
  alias SentientwaveAutomata.Temporal
  require Logger

  @reconcile_interval_ms 30_000

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  def reconcile do
    enabled_by_id =
      Agents.list_enabled_scheduled_tasks()
      |> Map.new(&{&1.id, &1})

    Enum.each(enabled_by_id, fn {_id, task} ->
      ensure_task_workflow(task)
    end)

    Agents.list_temporal_managed_scheduled_tasks()
    |> Enum.reject(&Map.has_key?(enabled_by_id, &1.id))
    |> Enum.each(fn task ->
      if is_binary(task.workflow_id) and task.workflow_id != "" do
        _ = temporal_adapter().signal_workflow(task.workflow_id, "stop", %{"task_id" => task.id})
      end
    end)

    :ok
  end

  @impl true
  def init(state) do
    send(self(), :reconcile)
    {:ok, state}
  end

  @impl true
  def handle_info(:reconcile, state) do
    reconcile()
    Process.send_after(self(), :reconcile, @reconcile_interval_ms)
    {:noreply, state}
  end

  defp ensure_task_workflow(%ScheduledTask{} = task) do
    desired_workflow_id = Temporal.child_workflow_id("scheduled_task", task.id)

    cond do
      task.workflow_id in [nil, ""] ->
        case temporal_adapter().start_workflow("scheduled_task_workflow", %{
               workflow_id: desired_workflow_id,
               task_id: task.id
             }) do
          {:ok, %{workflow_id: workflow_id, run_id: run_id}} ->
            _ =
              Agents.update_scheduled_task_temporal_state(task, %{
                workflow_id: workflow_id,
                temporal_run_id: run_id
              })

            :ok

          {:error, reason} ->
            Logger.warning(
              "scheduled_task_reconcile_start_failed task_id=#{task.id} reason=#{inspect(reason)}"
            )

            :ok
        end

      true ->
        _ =
          temporal_adapter().signal_workflow(task.workflow_id, "refresh", %{"task_id" => task.id})

        :ok
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
