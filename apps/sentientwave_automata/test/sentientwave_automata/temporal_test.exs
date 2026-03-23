defmodule SentientwaveAutomata.TemporalTest do
  use ExUnit.Case, async: true

  alias SentientwaveAutomata.Temporal

  setup do
    saved = %{
      cluster: Application.get_env(:sentientwave_automata, :temporal_cluster),
      namespace: Application.get_env(:sentientwave_automata, :temporal_namespace),
      workflow_queue: Application.get_env(:sentientwave_automata, :temporal_workflow_task_queue),
      activity_queue: Application.get_env(:sentientwave_automata, :temporal_activity_task_queue),
      worker_identity_prefix:
        Application.get_env(:sentientwave_automata, :temporal_worker_identity_prefix)
    }

    Application.put_env(:sentientwave_automata, :temporal_cluster, :temporal_test_cluster)
    Application.put_env(:sentientwave_automata, :temporal_namespace, "test-namespace")
    Application.put_env(:sentientwave_automata, :temporal_workflow_task_queue, "test-workflows")
    Application.put_env(:sentientwave_automata, :temporal_activity_task_queue, "test-activities")
    Application.put_env(:sentientwave_automata, :temporal_worker_identity_prefix, "test-worker")

    on_exit(fn ->
      restore_env(:temporal_cluster, saved.cluster)
      restore_env(:temporal_namespace, saved.namespace)
      restore_env(:temporal_workflow_task_queue, saved.workflow_queue)
      restore_env(:temporal_activity_task_queue, saved.activity_queue)
      restore_env(:temporal_worker_identity_prefix, saved.worker_identity_prefix)
    end)

    :ok
  end

  test "runtime helpers read the configured temporal cluster, namespace, and queues" do
    assert Temporal.cluster() == :temporal_test_cluster
    assert Temporal.namespace() == "test-namespace"
    assert Temporal.workflow_task_queue() == "test-workflows"
    assert Temporal.activity_task_queue() == "test-activities"
    assert Temporal.worker_identity_prefix() == "test-worker"
  end

  test "activity_payload stringifies nested keys before adding the step" do
    timestamp = DateTime.from_naive!(~N[2026-03-23 07:22:00], "Etc/UTC")

    payload = %{
      run_id: "run-123",
      attrs: %{
        room_id: "!dm:localhost",
        metadata: %{agent_slug: "automata"},
        tool_calls: [
          %{
            name: "system_directory_admin",
            arguments: %{action: "list_users"},
            issued_at: timestamp
          }
        ]
      }
    }

    assert Temporal.activity_payload("plan_tool_calls", payload) == %{
             "step" => "plan_tool_calls",
             "run_id" => "run-123",
             "attrs" => %{
               "room_id" => "!dm:localhost",
               "metadata" => %{"agent_slug" => "automata"},
               "tool_calls" => [
                 %{
                   "name" => "system_directory_admin",
                   "arguments" => %{"action" => "list_users"},
                   "issued_at" => timestamp
                 }
               ]
             }
           }
  end

  defp restore_env(_key, nil), do: :ok

  defp restore_env(key, value) do
    Application.put_env(:sentientwave_automata, key, value)
  end
end
