defmodule SentientwaveAutomata.Repo.Migrations.AddTemporalWorkflowCutoverSupport do
  use Ecto.Migration

  def change do
    create table(:orchestration_workflows, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :workflow_id, :string, null: false
      add :run_id, :string
      add :status, :string, null: false, default: "running"
      add :room_id, :string, null: false
      add :objective, :text, null: false
      add :requested_by, :string, null: false
      add :result, :map, null: false, default: %{}
      add :error, :map, null: false, default: %{}
      add :metadata, :map, null: false, default: %{}
      timestamps(type: :utc_datetime_usec)
    end

    create unique_index(:orchestration_workflows, [:workflow_id])
    create index(:orchestration_workflows, [:status])
    create index(:orchestration_workflows, [:inserted_at])

    alter table(:agent_scheduled_tasks) do
      add :workflow_id, :string
      add :temporal_run_id, :string
    end

    create unique_index(:agent_scheduled_tasks, [:workflow_id])
  end
end
