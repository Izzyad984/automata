defmodule SentientwaveAutomata.Temporal do
  @moduledoc """
  Shared Temporal runtime configuration helpers.
  """

  @default_cluster :automata
  @default_namespace "default"
  @default_workflow_task_queue "automata-workflows"
  @default_activity_task_queue "automata-activities"

  def cluster do
    Application.get_env(:sentientwave_automata, :temporal_cluster, @default_cluster)
  end

  def namespace do
    Application.get_env(:sentientwave_automata, :temporal_namespace, @default_namespace)
  end

  def workflow_task_queue do
    Application.get_env(
      :sentientwave_automata,
      :temporal_workflow_task_queue,
      @default_workflow_task_queue
    )
  end

  def activity_task_queue do
    Application.get_env(
      :sentientwave_automata,
      :temporal_activity_task_queue,
      @default_activity_task_queue
    )
  end

  def worker_identity_prefix do
    Application.get_env(
      :sentientwave_automata,
      :temporal_worker_identity_prefix,
      "automata"
    )
  end

  def workflow_execution(workflow_id, run_id \\ nil) when is_binary(workflow_id) do
    execution = %{workflow_id: workflow_id}

    case normalize_string(run_id) do
      nil -> execution
      value -> Map.put(execution, :run_id, value)
    end
  end

  def generated_workflow_id(prefix) when is_binary(prefix) do
    base = prefix |> String.trim() |> String.replace(~r/[^a-zA-Z0-9._-]+/u, "_")
    "#{base}_#{Ecto.UUID.generate()}"
  end

  def child_workflow_id(prefix, stable_id) do
    normalized_prefix = prefix |> String.trim() |> String.replace(~r/[^a-zA-Z0-9._-]+/u, "_")
    normalized_id = stable_id |> to_string() |> String.replace(~r/[^a-zA-Z0-9._-]+/u, "_")
    "#{normalized_prefix}_#{normalized_id}"
  end

  def normalize_string(value) when is_binary(value) do
    case String.trim(value) do
      "" -> nil
      trimmed -> trimmed
    end
  end

  def normalize_string(value) when is_atom(value),
    do: value |> Atom.to_string() |> normalize_string()

  def normalize_string(_value), do: nil
end
