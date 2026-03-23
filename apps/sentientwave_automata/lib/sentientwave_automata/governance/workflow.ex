defmodule SentientwaveAutomata.Governance.Workflow do
  @moduledoc """
  Control-plane boundary for Temporal-owned governance proposal workflows.
  """

  require Logger

  alias SentientwaveAutomata.Governance
  alias SentientwaveAutomata.Governance.LawProposal
  alias SentientwaveAutomata.Matrix.Directory
  alias SentientwaveAutomata.Matrix.DirectoryUser
  alias SentientwaveAutomata.Repo
  alias SentientwaveAutomata.Temporal

  @poll_interval_ms 100
  @poll_timeout_ms 3_000
  @vote_signal "vote"
  @resolve_signal "resolve"

  @spec handle_command(map()) :: {:ok, map()} | {:error, term()} | :ignore
  def handle_command(%{proposal_type: _} = command), do: open_proposal(command)
  def handle_command(%{choice: _} = command), do: cast_vote(command)
  def handle_command(_command), do: :ignore

  @spec open_proposal(map()) :: {:ok, LawProposal.t()} | {:error, term()}
  def open_proposal(command) when is_map(command) do
    with {:ok, actor} <- resolve_actor(command),
         true <- allowed_to_open?(actor) || {:error, :not_authorized},
         workflow_id <- Temporal.generated_workflow_id("governance_proposal"),
         {:ok, _temporal} <-
           temporal_adapter().start_workflow(
             "governance_proposal_workflow",
             %{
               "mode" => "open",
               "workflow_id" => workflow_id,
               "command" => normalize_command(command)
             },
             workflow_id: workflow_id
           ),
         {:ok, proposal} <- await_proposal(workflow_id, @poll_timeout_ms) do
      {:ok, proposal}
    end
  end

  @spec cast_vote(map()) :: {:ok, Governance.LawVote.t()} | {:error, term()}
  def cast_vote(command) when is_map(command) do
    with {:ok, actor} <- resolve_actor(command),
         {:ok, proposal} <- resolve_open_proposal(command),
         {:ok, workflow_id} <- ensure_proposal_workflow(proposal),
         :ok <-
           temporal_adapter().signal_workflow(
             workflow_id,
             @vote_signal,
             normalize_command(Map.put(command, :actor_id, actor.id))
           ),
         {:ok, vote} <- await_vote(proposal.id, actor.id, @poll_timeout_ms) do
      {:ok, vote}
    end
  end

  @spec resolve_proposal(map() | binary()) :: {:ok, LawProposal.t()} | {:error, term()}
  def resolve_proposal(reference) when is_binary(reference) do
    resolve_proposal(%{"reference" => reference})
  end

  def resolve_proposal(%{} = attrs) do
    with {:ok, proposal} <- resolve_proposal_record(attrs),
         {:ok, workflow_id} <- ensure_proposal_workflow(proposal),
         :ok <-
           temporal_adapter().signal_workflow(
             workflow_id,
             @resolve_signal,
             normalize_command(attrs)
           ),
         {:ok, resolved} <- await_resolved_proposal(proposal.id, @poll_timeout_ms) do
      {:ok, resolved}
    end
  end

  @spec current_constitution_snapshot() :: Governance.ConstitutionSnapshot.t() | nil
  def current_constitution_snapshot, do: Governance.current_constitution_snapshot()

  @spec proposal_results(Governance.LawProposal.t() | binary()) :: map() | {:error, term()}
  def proposal_results(%LawProposal{} = proposal), do: Governance.proposal_results(proposal)

  def proposal_results(reference) when is_binary(reference) do
    case Governance.get_proposal_by_reference(reference) do
      %LawProposal{} = proposal -> Governance.proposal_results(proposal)
      nil -> {:error, :not_found}
    end
  end

  @spec reconcile_open_proposals() :: :ok
  def reconcile_open_proposals do
    Governance.list_proposals(status: :open)
    |> Enum.each(fn proposal ->
      case ensure_proposal_workflow(proposal) do
        {:ok, _workflow_id} ->
          :ok

        {:error, reason} ->
          Logger.warning(
            "governance_temporal_reconcile_failed proposal_id=#{proposal.id} reference=#{proposal.reference} reason=#{inspect(reason)}"
          )
      end
    end)

    :ok
  end

  defp ensure_proposal_workflow(%LawProposal{} = proposal) do
    workflow_id =
      proposal.workflow_id ||
        Temporal.child_workflow_id("governance_proposal", proposal.id)

    case proposal.workflow_id do
      value when is_binary(value) and value != "" ->
        case temporal_adapter().query_workflow(value) do
          {:ok, _status} ->
            {:ok, value}

          {:error, _reason} ->
            start_resume_workflow(proposal, workflow_id)
        end

      _ ->
        start_resume_workflow(proposal, workflow_id)
    end
  end

  defp start_resume_workflow(%LawProposal{} = proposal, workflow_id) do
    case temporal_adapter().start_workflow(
           "governance_proposal_workflow",
           %{
             "mode" => "resume",
             "workflow_id" => workflow_id,
             "proposal_id" => proposal.id
           },
           workflow_id: workflow_id
         ) do
      {:ok, _temporal} ->
        proposal
        |> Ecto.Changeset.change(%{workflow_id: workflow_id})
        |> Repo.update()
        |> case do
          {:ok, updated} -> {:ok, updated.workflow_id}
          {:error, reason} -> {:error, reason}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp await_proposal(_workflow_id, remaining_ms) when remaining_ms <= 0,
    do: {:error, :proposal_not_persisted}

  defp await_proposal(workflow_id, remaining_ms) do
    case Repo.get_by(LawProposal, workflow_id: workflow_id) do
      %LawProposal{} = proposal ->
        {:ok, Governance.get_proposal(proposal)}

      nil ->
        Process.sleep(@poll_interval_ms)
        await_proposal(workflow_id, remaining_ms - @poll_interval_ms)
    end
  end

  defp await_vote(_proposal_id, _actor_id, remaining_ms) when remaining_ms <= 0,
    do: {:error, :vote_not_persisted}

  defp await_vote(proposal_id, actor_id, remaining_ms) do
    case Repo.get_by(Governance.LawVote, proposal_id: proposal_id, voter_id: actor_id) do
      %Governance.LawVote{} = vote ->
        {:ok, Repo.preload(vote, [:voter])}

      nil ->
        Process.sleep(@poll_interval_ms)
        await_vote(proposal_id, actor_id, remaining_ms - @poll_interval_ms)
    end
  end

  defp await_resolved_proposal(_proposal_id, remaining_ms) when remaining_ms <= 0,
    do: {:error, :proposal_not_resolved}

  defp await_resolved_proposal(proposal_id, remaining_ms) do
    case Governance.get_proposal(proposal_id) do
      %LawProposal{status: :open} ->
        Process.sleep(@poll_interval_ms)
        await_resolved_proposal(proposal_id, remaining_ms - @poll_interval_ms)

      %LawProposal{} = proposal ->
        {:ok, proposal}

      nil ->
        {:error, :not_found}
    end
  end

  defp resolve_open_proposal(command) do
    with {:ok, proposal} <- resolve_proposal_record(command),
         true <- proposal.status == :open || {:error, :proposal_closed} do
      {:ok, proposal}
    end
  end

  defp resolve_proposal_record(attrs) when is_map(attrs) do
    cond do
      is_binary(fetch_value(attrs, "proposal_id")) ->
        case Governance.get_proposal(fetch_value(attrs, "proposal_id")) do
          %LawProposal{} = proposal -> {:ok, proposal}
          nil -> {:error, :not_found}
        end

      is_binary(fetch_value(attrs, "reference")) ->
        case Governance.get_proposal_by_reference(fetch_value(attrs, "reference")) do
          %LawProposal{} = proposal -> {:ok, proposal}
          nil -> {:error, :not_found}
        end

      true ->
        {:error, :missing_reference}
    end
  end

  defp resolve_actor(command) do
    sender_mxid = fetch_value(command, "sender_mxid")

    with true <-
           (is_binary(sender_mxid) and String.trim(sender_mxid) != "") ||
             {:error, :missing_sender},
         localpart <-
           sender_mxid |> String.trim_leading("@") |> String.split(":", parts: 2) |> List.first(),
         %DirectoryUser{} = actor <- Directory.get_user_record(localpart) do
      {:ok, actor}
    else
      nil -> {:error, :unknown_sender}
      {:error, reason} -> {:error, reason}
      _ -> {:error, :unknown_sender}
    end
  end

  defp allowed_to_open?(%DirectoryUser{admin: true}), do: true
  defp allowed_to_open?(%DirectoryUser{kind: :person}), do: true
  defp allowed_to_open?(_actor), do: false

  defp normalize_command(command) when is_map(command) do
    Enum.reduce(command, %{}, fn {key, value}, acc ->
      normalized_key =
        case key do
          atom when is_atom(atom) -> Atom.to_string(atom)
          binary when is_binary(binary) -> binary
        end

      Map.put(acc, normalized_key, normalize_value(value))
    end)
  end

  defp normalize_value(value) when is_map(value), do: normalize_command(value)
  defp normalize_value(value) when is_list(value), do: Enum.map(value, &normalize_value/1)
  defp normalize_value(value), do: value

  defp fetch_value(map, key) do
    atom_key =
      case key do
        "proposal_id" -> :proposal_id
        "reference" -> :reference
        "sender_mxid" -> :sender_mxid
        _ -> nil
      end

    Map.get(map, key) || (atom_key && Map.get(map, atom_key))
  end

  defp temporal_adapter do
    Application.get_env(
      :sentientwave_automata,
      :temporal_adapter,
      SentientwaveAutomata.Adapters.Temporal.Runtime
    )
  end
end
