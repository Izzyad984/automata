defmodule SentientwaveAutomata.Agents.WorkflowActivities do
  @moduledoc """
  Temporal activity entrypoint for agent run execution steps.
  """

  use TemporalSdk.Activity

  alias SentientwaveAutomata.Agents
  alias SentientwaveAutomata.Agents.Activities
  alias SentientwaveAutomata.Agents.LLM.Client
  alias SentientwaveAutomata.Agents.Run
  require Logger

  @impl true
  def execute(_context, [%{"step" => "build_context", "run_id" => run_id, "attrs" => attrs}]) do
    run = fetch_run!(run_id)
    [unwrap_result!(Activities.build_context(run, attrs), "build agent context")]
  end

  def execute(_context, [%{"step" => "compact_context", "run_id" => run_id, "context" => context}]) do
    run = fetch_run!(run_id)
    [unwrap_result!(Activities.compact_context(run, context), "compact agent context")]
  end

  def execute(
        _context,
        [
          %{
            "step" => "plan_tool_calls",
            "run_id" => run_id,
            "attrs" => attrs,
            "context" => workflow_context
          }
        ]
      ) do
    run = fetch_run!(run_id)

    with_typing_lease(run, attrs, fn ->
      [
        Client.plan_tool_calls(tool_client_opts(run, attrs, workflow_context))
        |> unwrap_result!("plan tool calls")
        |> then(fn plan -> Map.get(plan, :tool_calls) || Map.get(plan, "tool_calls") || [] end)
      ]
    end)
  end

  def execute(_context, [
        %{"step" => "execute_tool_calls", "run_id" => run_id, "tool_calls" => tool_calls}
      ]) do
    run = fetch_run!(run_id)
    [unwrap_result!(Client.execute_tool_calls(run.agent_id, tool_calls), "execute tool calls")]
  end

  def execute(
        _context,
        [
          %{
            "step" => "synthesize_response",
            "run_id" => run_id,
            "attrs" => attrs,
            "context" => workflow_context,
            "tool_context" => tool_context
          }
        ]
      ) do
    run = fetch_run!(run_id)

    with_typing_lease(run, attrs, fn ->
      [
        Client.synthesize_tool_response(
          tool_client_opts(run, attrs, workflow_context),
          tool_context
        )
        |> unwrap_result!("synthesize tool response")
      ]
    end)
  end

  def execute(
        _context,
        [
          %{
            "step" => "generate_response_without_tools",
            "run_id" => run_id,
            "attrs" => attrs,
            "context" => workflow_context
          }
        ]
      ) do
    run = fetch_run!(run_id)

    with_typing_lease(run, attrs, fn ->
      [
        Client.generate_response_without_tools(tool_client_opts(run, attrs, workflow_context))
        |> unwrap_result!("generate agent response")
      ]
    end)
  end

  def execute(
        _context,
        [
          %{
            "step" => "post_response",
            "run_id" => run_id,
            "attrs" => attrs,
            "response" => response
          }
        ]
      ) do
    run = fetch_run!(run_id)

    :ok =
      unwrap_result!(Activities.post_response(run, attrs, response), "post response to Matrix")

    [%{"posted" => true}]
  end

  def execute(
        _context,
        [
          %{
            "step" => "persist_memory",
            "run_id" => run_id,
            "attrs" => attrs,
            "context" => workflow_context,
            "response" => response
          }
        ]
      ) do
    run = fetch_run!(run_id)
    :ok = Activities.persist_memory(run, attrs, workflow_context, response)
    [%{"persisted" => true}]
  end

  def execute(
        _context,
        [
          %{
            "step" => "mark_run_status",
            "run_id" => run_id,
            "status" => status,
            "updates" => updates
          }
        ]
      ) do
    run = fetch_run!(run_id)

    updated_run =
      unwrap_result!(Agents.update_run(run, Map.put(updates, :status, status)), "update run")

    [%{"run_id" => updated_run.id, "status" => Atom.to_string(updated_run.status)}]
  end

  def execute(_context, [
        %{
          "step" => "set_typing",
          "room_id" => room_id,
          "typing" => typing,
          "metadata" => metadata
        }
      ]) do
    case set_typing(room_id, typing, metadata) do
      :ok -> [%{"typing" => typing}]
      {:error, reason} -> raise "failed to update typing state: #{inspect(reason)}"
    end
  end

  def execute(_context, [payload]) do
    raise "unsupported agent workflow activity step: #{inspect(payload)}"
  end

  defp tool_client_opts(%Run{} = run, attrs, workflow_context) do
    input = fetch_map(attrs, "input")
    metadata = fetch_map(attrs, "metadata")

    [
      agent_id: run.agent_id,
      agent_slug: fetch_value(metadata, "agent_slug", "automata"),
      user_input: fetch_value(input, "body", ""),
      context_text: fetch_value(workflow_context, "context_text", ""),
      room_id: fetch_value(attrs, "room_id", ""),
      constitution_snapshot:
        Map.get(run.metadata || %{}, "constitution_snapshot_id") &&
          %{
            id: Map.get(run.metadata || %{}, "constitution_snapshot_id"),
            version: Map.get(run.metadata || %{}, "constitution_version")
          },
      trace_context: %{
        run_id: run.id,
        room_id: fetch_value(attrs, "room_id", ""),
        requested_by: fetch_value(attrs, "requested_by"),
        remote_ip: fetch_value(attrs, "remote_ip"),
        conversation_scope: fetch_value(attrs, "conversation_scope")
      }
    ]
  end

  defp fetch_run!(run_id) when is_binary(run_id) do
    case Agents.get_run(run_id) do
      %Run{} = run -> run
      nil -> raise "run not found: #{run_id}"
    end
  end

  defp fetch_run!(_run_id), do: raise("run not found")

  defp with_typing_lease(%Run{} = run, attrs, fun) when is_function(fun, 0) do
    room_id = fetch_value(attrs, "room_id", "")
    metadata = %{run_id: run.id, workflow_id: run.workflow_id}

    heartbeat_pid =
      if is_binary(room_id) and String.trim(room_id) != "" do
        start_typing_heartbeat(room_id, metadata)
      else
        nil
      end

    try do
      fun.()
    after
      stop_typing_heartbeat(heartbeat_pid)
      _ = set_typing(room_id, false, metadata)
    end
  end

  defp start_typing_heartbeat(room_id, metadata) do
    parent = self()

    spawn_link(fn ->
      typing_loop(parent, room_id, metadata, typing_interval_ms())
    end)
  end

  defp stop_typing_heartbeat(nil), do: :ok

  defp stop_typing_heartbeat(pid) when is_pid(pid) do
    ref = make_ref()
    send(pid, {:stop, self(), ref})

    receive do
      {:stopped, ^ref} -> :ok
    after
      500 -> :ok
    end
  end

  defp typing_loop(parent, room_id, metadata, interval_ms) do
    _ = set_typing(room_id, true, metadata)

    receive do
      {:stop, caller, ref} ->
        send(caller, {:stopped, ref})
        :ok
    after
      interval_ms ->
        if Process.alive?(parent) do
          typing_loop(parent, room_id, metadata, interval_ms)
        else
          :ok
        end
    end
  end

  defp set_typing(room_id, typing, metadata) when is_binary(room_id) and room_id != "" do
    matrix_adapter().set_typing(room_id, typing, typing_timeout_ms(), metadata)
  end

  defp set_typing(_room_id, _typing, _metadata), do: :ok

  defp matrix_adapter do
    Application.get_env(
      :sentientwave_automata,
      :matrix_adapter,
      SentientwaveAutomata.Adapters.Matrix.Local
    )
  end

  defp typing_interval_ms do
    max(div(typing_timeout_ms(), 2), 1_000)
  end

  defp typing_timeout_ms do
    System.get_env("MATRIX_TYPING_TIMEOUT_MS", "12000")
    |> String.to_integer()
  rescue
    _ -> 12_000
  end

  defp unwrap_result!({:ok, result}, _action), do: result

  defp unwrap_result!({:error, reason}, action) do
    raise "#{action} failed: #{inspect(reason)}"
  end

  defp unwrap_result!(result, _action), do: result

  defp fetch_map(map, key) do
    case fetch_value(map, key) do
      value when is_map(value) -> value
      _ -> %{}
    end
  end

  defp fetch_value(map, key, default \\ nil) when is_map(map) do
    atom_key =
      case key do
        "input" -> :input
        "metadata" -> :metadata
        "agent_slug" -> :agent_slug
        "body" -> :body
        "context_text" -> :context_text
        "room_id" -> :room_id
        "requested_by" -> :requested_by
        "remote_ip" -> :remote_ip
        "conversation_scope" -> :conversation_scope
        _ -> nil
      end

    Map.get(map, key, atom_key && Map.get(map, atom_key, default)) || default
  end
end
