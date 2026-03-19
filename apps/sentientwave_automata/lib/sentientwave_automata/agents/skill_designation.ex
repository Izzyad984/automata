defmodule SentientwaveAutomata.Agents.SkillDesignation do
  use Ecto.Schema
  import Ecto.Changeset

  @statuses [:active, :rolled_back]

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "agent_skill_designations" do
    field :status, Ecto.Enum, values: @statuses, default: :active
    field :designated_at, :utc_datetime_usec
    field :rolled_back_at, :utc_datetime_usec
    field :metadata, :map, default: %{}

    belongs_to :skill, SentientwaveAutomata.Agents.Skill
    belongs_to :agent, SentientwaveAutomata.Agents.AgentProfile

    timestamps(type: :utc_datetime_usec)
  end

  def statuses, do: @statuses

  def changeset(designation, attrs) do
    designation
    |> cast(attrs, [:skill_id, :agent_id, :status, :designated_at, :rolled_back_at, :metadata])
    |> validate_required([:skill_id, :agent_id, :status, :designated_at])
    |> assoc_constraint(:skill)
    |> assoc_constraint(:agent)
    |> unique_constraint(:skill_id,
      name: :agent_skill_designations_active_skill_agent_index
    )
  end
end
