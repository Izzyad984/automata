defmodule SentientwaveAutomata.Orchestration.Activities do
  @moduledoc """
  Temporal activity entrypoint for generic conversation workflows.
  """

  use TemporalSdk.Activity

  alias SentientwaveAutomata.Orchestration.Workflow
  alias SentientwaveAutomata.Repo

  @impl true
  def execute(
        _context,
        [%{"step" => "post_started_message", "workflow_id" => workflow_id, "attrs" => attrs}]
      ) do
    room_id = fetch_value(attrs, "room_id")
    objective = fetch_value(attrs, "objective")
    requested_by = fetch_value(attrs, "requested_by")

    if room_id in [nil, ""] do
      [%{"posted" => false}]
    else
      case matrix_adapter().post_message(room_id, "Workflow started: #{objective}", %{
             "workflow_id" => workflow_id,
             "requested_by" => requested_by,
             "kind" => "conversation_workflow_started"
           }) do
        :ok -> [%{"posted" => true, "room_id" => room_id}]
        {:error, reason} -> raise "failed to post started message: #{inspect(reason)}"
      end
    end
  end

  def execute(
        _context,
        [
          %{
            "step" => "mark_status",
            "workflow_id" => workflow_id,
            "status" => status
          } = payload
        ]
      ) do
    case Repo.get_by(Workflow, workflow_id: workflow_id) do
      %Workflow{} = workflow ->
        attrs = %{
          status: normalize_status(status),
          result: Map.get(payload, "result", %{}),
          error: Map.get(payload, "error", %{})
        }

        case workflow |> Workflow.changeset(attrs) |> Repo.update() do
          {:ok, updated} ->
            [
              %{
                "workflow_id" => updated.workflow_id,
                "status" => Atom.to_string(updated.status)
              }
            ]

          {:error, reason} ->
            raise "failed to mark orchestration workflow status: #{inspect(reason)}"
        end

      nil ->
        raise "workflow not found: #{workflow_id}"
    end
  end

  def execute(_context, [payload]) do
    raise "unsupported orchestration activity step: #{inspect(payload)}"
  end

  defp fetch_value(map, key) when is_map(map) do
    atom_key =
      case key do
        "room_id" -> :room_id
        "objective" -> :objective
        "requested_by" -> :requested_by
        _ -> nil
      end

    Map.get(map, key) || (atom_key && Map.get(map, atom_key))
  end

  defp normalize_status(status) when is_atom(status), do: status

  defp normalize_status(status) when is_binary(status) do
    case String.trim(status) do
      "running" -> :running
      "succeeded" -> :succeeded
      "failed" -> :failed
      "cancelled" -> :cancelled
      _ -> :running
    end
  end

  defp matrix_adapter do
    Application.get_env(
      :sentientwave_automata,
      :matrix_adapter,
      SentientwaveAutomata.Adapters.Matrix.Local
    )
  end
end
