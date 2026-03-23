defmodule SentientwaveAutomata.Orchestration.Workflow do
  use Ecto.Schema
  import Ecto.Changeset

  @statuses [:running, :succeeded, :failed, :cancelled]

  @primary_key {:id, :binary_id, autogenerate: true}
  schema "orchestration_workflows" do
    field :workflow_id, :string
    field :run_id, :string
    field :status, Ecto.Enum, values: @statuses, default: :running
    field :room_id, :string
    field :objective, :string
    field :requested_by, :string
    field :result, :map, default: %{}
    field :error, :map, default: %{}
    field :metadata, :map, default: %{}

    timestamps(type: :utc_datetime_usec)
  end

  def statuses, do: @statuses

  def changeset(workflow, attrs) do
    workflow
    |> cast(attrs, [
      :workflow_id,
      :run_id,
      :status,
      :room_id,
      :objective,
      :requested_by,
      :result,
      :error,
      :metadata
    ])
    |> validate_required([:workflow_id, :status, :room_id, :objective, :requested_by])
    |> unique_constraint(:workflow_id)
  end
end
